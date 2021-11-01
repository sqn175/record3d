#include <iostream>
#include <vector>
#include <record3d/Record3DStream.h>
#include <mutex>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <cmath>

#ifdef HAS_OPENCV
#include <opencv2/opencv.hpp>
#endif

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wmissing-noreturn"

using namespace std;


/**
 * A simple demo app presenting how to use the Record3D library to display RGBD stream.
 */
class Record3DDemoApp
{
public:
    Record3DDemoApp(const std::string& base_path_string)
    {
        const auto p1 = std::chrono::system_clock::now();
        auto time = std::chrono::duration_cast<std::chrono::seconds>(p1.time_since_epoch()).count();
        path_str_ = base_path_string + "/" + std::to_string(time) + "/";
        std::cout << "Dataset save path: " << path_str_ << std::endl;
        const std::filesystem::path path{path_str_};
        std::filesystem::remove_all(path);
        std::filesystem::create_directories(path/"rgbd");
        std::filesystem::create_directories(path/"depth");
    }

    void Run()
    {
        Record3D::Record3DStream stream{};
        stream.onStreamStopped = [&]
        {
            OnStreamStopped();
        };
        stream.onNewFrame = [&](const Record3D::BufferRGB &$rgbFrame,
                                const Record3D::BufferDepth &$depthFrame,
                                uint32_t $rgbWidth,
                                uint32_t $rgbHeight,
                                uint32_t $depthWidth,
                                uint32_t $depthHeight,
                                Record3D::DeviceType $deviceType,
                                Record3D::IntrinsicMatrixCoeffs $K)
        {
            OnNewFrame( $rgbFrame, $depthFrame, $rgbWidth, $rgbHeight, $depthWidth, $depthHeight, $deviceType, $K );
        };

        // Try connecting to a device.
        const auto &devs = Record3D::Record3DStream::GetConnectedDevices();
        if ( devs.empty())
        {
            fprintf( stderr,
                     "No iOS devices found. Ensure you have connected your iDevice via USB to this computer.\n" );
            return;
        }
        else
        {
            printf( "Found %lu iOS device(s):\n", devs.size());
            for ( const auto &dev : devs )
            {
                printf( "\tDevice ID: %u\n\tUDID: %s\n\n", dev.productId, dev.udid.c_str());
            }
        }

        const auto &selectedDevice = devs[ 0 ];
        printf( "Trying to connect to device with ID %u.\n", selectedDevice.productId );

        bool isConnected = stream.ConnectToDevice( devs[ 0 ] );
        if ( isConnected )
        {
            printf( "Connected and starting to stream. Enable USB streaming in the Record3D iOS app (https://record3d.app/) in case you don't see RGBD stream.\n" );
            while ( true )
            {
                // Wait for the callback thread to receive new frame and unlock this thread
#ifdef HAS_OPENCV
                if ( imgRGB_.cols == 0 || imgRGB_.rows == 0 || imgDepth_.cols == 0 || imgDepth_.rows == 0 )
                {
                    continue;
                }

                cv::Mat rgb, depth;
                {
                    std::lock_guard<std::recursive_mutex> lock(mainThreadLock_);
                    rgb = imgRGB_.clone();
                    depth = imgDepth_.clone();
                }
                // Postprocess images
                cv::cvtColor( rgb, rgb, cv::COLOR_RGB2BGR );

                // The TrueDepth camera is a selfie camera; we mirror the RGBD frame so it looks plausible.
                if ( currentDeviceType_ == Record3D::R3D_DEVICE_TYPE__FACEID )
                {
                    cv::flip( rgb, rgb, 1 );
                    cv::flip( depth, depth, 1 );
                }

                // Show images
                cv::imshow( "RGB", rgb );
                cv::imshow( "Depth", depth );
                cv::waitKey( 1 );
#endif
            }
        }
        else
        {
            fprintf( stderr,
                     "Could not connect to iDevice. Make sure you have opened the Record3D iOS app (https://record3d.app/).\n" );
        }
    }

private:
    void OnStreamStopped()
    {
        fprintf( stderr, "Stream stopped!" );
    }

    void OnNewFrame(const Record3D::BufferRGB &$rgbFrame,
                    const Record3D::BufferDepth &$depthFrame,
                    uint32_t $rgbWidth,
                    uint32_t $rgbHeight,
                    uint32_t $depthWidth,
                    uint32_t $depthHeight,
                    Record3D::DeviceType $deviceType,
                    Record3D::IntrinsicMatrixCoeffs $K)
    {
        currentDeviceType_ = (Record3D::DeviceType) $deviceType;

        if (!intrinsic_matrix_saved_)
        {
            std::ofstream calib(path_str_ + "calibration.txt", std::ofstream::out);
            calib << $K.fx << " " << $K.fy << " " << $K.tx << " " << $K.ty; 
            calib.close();
            intrinsic_matrix_saved_ = true;
        }

#ifdef HAS_OPENCV
        std::lock_guard<std::recursive_mutex> lock(mainThreadLock_);
        // When we switch between the TrueDepth and the LiDAR camera, the size frame size changes.
        // Recreate the RGB and Depth images with fitting size.
        if (    imgRGB_.rows != $rgbHeight || imgRGB_.cols != $rgbWidth
             || imgDepth_.rows != $depthHeight || imgDepth_.cols != $depthWidth )
        {
            imgRGB_.release();
            imgDepth_.release();

            imgRGB_ = cv::Mat::zeros( $rgbHeight, $rgbWidth, CV_8UC3);
            imgDepth_ = cv::Mat::zeros( $depthHeight, $depthWidth, CV_32F );
        }

        // The `BufferRGB` and `BufferDepth` may be larger than the actual payload, therefore the true frame size is computed.
        constexpr int numRGBChannels = 3;
        memcpy( imgRGB_.data, $rgbFrame.data(), $rgbWidth * $rgbHeight * numRGBChannels * sizeof(uint8_t));
        memcpy( imgDepth_.data, $depthFrame.data(), $depthWidth * $depthHeight * sizeof(float));

        // Save images
        // Save depth map as ETH3D dataset described.
        cv::Mat imgScaledDepth($depthHeight, $depthWidth, CV_16U);
        for (int i = 0; i < imgDepth_.rows; ++i)
        {
            for (int j = 0; j < imgDepth_.cols; ++j)
            {
                imgScaledDepth.at<uint16_t>(i,j) = std::round(imgDepth_.at<float>(i,j)*5000);
            }
        }
        const auto p1 = std::chrono::system_clock::now();
        auto time = std::chrono::duration_cast<std::chrono::microseconds>(p1.time_since_epoch()).count();
        const std::string file_name = std::to_string(time / 1000000) + "." + std::to_string(time % 1000000) + ".png";
        if (!(cv::imwrite(path_str_ + "rgb/" + file_name, imgRGB_) && cv::imwrite(path_str_ + "depth/" + file_name, imgScaledDepth)))
        {
            std::cout << "Saving image failed!" << std::endl;
        }
#endif
    }

private:
    std::recursive_mutex mainThreadLock_{};
    Record3D::DeviceType currentDeviceType_{};
    std::string path_str_{"./"};
    bool intrinsic_matrix_saved_ = false;
    
#ifdef HAS_OPENCV
    cv::Mat imgRGB_{};
    cv::Mat imgDepth_{};
#endif
};


int main(int argc, char** argv)
{
    std::string base_path_string = argc > 1 ? std::string{argv[1]} : "./";
    Record3DDemoApp app(base_path_string);
    app.Run();
}

#pragma clang diagnostic pop
