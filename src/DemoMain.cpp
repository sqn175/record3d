#include <iostream>
#include <vector>
#include <record3d/Record3DStream.h>
#include <mutex>
#include <condition_variable>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <cmath>

#include <signal.h>

#ifdef HAS_OPENCV
#include <opencv2/opencv.hpp>
#endif

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wmissing-noreturn"

using namespace std;

std::atomic<bool> is_quit {false};

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
        std::filesystem::create_directories(path/"rgb");
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
            std::ofstream associated(path_str_ + "associated.txt", std::ofstream::out);
            printf( "Connected and starting to stream. Enable USB streaming in the Record3D iOS app (https://record3d.app/) in case you don't see RGBD stream.\n" );
            while ( !is_quit.load(std::memory_order_acquire) )
            {
#ifdef HAS_OPENCV
                std::unique_lock<std::mutex> lock(mutex_);
                // Wait for the callback thread to receive new frame and unlock this thread
                data_ready_cv_.wait(lock);
                if ( imgRGB_.cols == 0 || imgRGB_.rows == 0 || imgDepth_.cols == 0 || imgDepth_.rows == 0 )
                {
                    continue;
                }
                cv::Mat rgb;
                rgb = imgRGB_.clone();
                // Transform depth map as ETH3D dataset described.
                cv::Mat imgScaledDepth(imgDepth_.rows, imgDepth_.cols, CV_16U);
                for (int i = 0; i < imgDepth_.rows; ++i)
                {
                    for (int j = 0; j < imgDepth_.cols; ++j)
                    {
                        imgScaledDepth.at<uint16_t>(i,j) = std::round(imgDepth_.at<float>(i,j)*5000);
                    }
                }
                
                // // Postprocess images
                // cv::cvtColor( rgb, rgb, cv::COLOR_RGB2BGR );

                // The TrueDepth camera is a selfie camera; we mirror the RGBD frame so it looks plausible.
                if ( currentDeviceType_ == Record3D::R3D_DEVICE_TYPE__FACEID )
                {
                    cv::flip( rgb, rgb, 1 );
                    cv::flip( imgScaledDepth, imgScaledDepth, 1 );
                }

                // Show images
                cv::imshow( "RGB", rgb );
                // cv::imshow( "Depth", imgScaledDepth );
                cv::waitKey( 1 );

                auto time = std::chrono::duration_cast<std::chrono::microseconds>(time_recv_.time_since_epoch()).count();
                const std::string time_str = std::to_string(time / 1000000) + "." + std::to_string(time % 1000000);
                const std::string file_name = time_str + ".png";
                // Save images
                try
                {
                    cv::imwrite(path_str_ + "rgb/" + file_name, rgb);
                    cv::imwrite(path_str_ + "depth/" + file_name, imgScaledDepth);
                }
                catch(cv::Exception& ex)
                {
                    std::cerr << "save image failed: " << ex.what() << '\n';
                }
                associated << time_str << " " << "rgb/" + file_name << " " << time_str  << " " << "depth/" + file_name << std::endl;

                // Calculate frequency
                auto time_now = std::chrono::system_clock::now();
                auto dur = std::chrono::duration_cast<std::chrono::milliseconds>(time_now - time_start_).count();
                int hz = 1000 / (dur / recv_cnt_);
                if (recv_cnt_ % 10 == 0)
                {
                    std::cout << "Image frequency (hz): " << hz <<  std::endl;
                }
#else
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
#endif
            }
            associated.close();

        }
        else
        {
            fprintf( stderr,
                     "Could not connect to iDevice. Make sure you have opened the Record3D iOS app (https://record3d.app/).\n" );
        }
        stream.Disconnect();
    }

    void Exit()
    {
        is_quit.store(true, std::memory_order_release);
        data_ready_cv_.notify_all(); 
    }

private:
    void OnStreamStopped()
    {
        fprintf( stderr, "Stream stopped!\n" );
        auto time_now = std::chrono::system_clock::now();
        auto dur = std::chrono::duration_cast<std::chrono::milliseconds>(time_now - time_start_).count();
        std::cout << "Record " << recv_cnt_ << " images, time length (ms): " << dur << std::endl;\
        is_quit.store(true, std::memory_order_release);
        data_ready_cv_.notify_all(); 
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
            time_start_ = std::chrono::system_clock::now();
            std::ofstream calib(path_str_ + "calibration.txt", std::ofstream::out);
            calib << $K.fx << " " << $K.fy << " " << $K.tx << " " << $K.ty; 
            calib.close();
            intrinsic_matrix_saved_ = true;
        }

#ifdef HAS_OPENCV
        std::lock_guard<std::mutex> lock(mutex_);
        // Calculate frequency
        auto time_now = std::chrono::system_clock::now();
        auto dur = std::chrono::duration_cast<std::chrono::milliseconds>(time_now - time_recv_).count();
        int hz_tmp = 1000 / dur;
        if (++recv_cnt_ % 10 == 0)
        {
            std::cout << "Instant frequency (ms, hz): " << dur << ", " << hz_tmp << std::endl;
        }
        time_recv_ = time_now;

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

        // When the RGB and Depth images are ready, notify other threads to process, e.g., visualize or save
        data_ready_cv_.notify_all(); 
#endif
    }

private:
    std::mutex mutex_{};
    std::condition_variable data_ready_cv_{};
    Record3D::DeviceType currentDeviceType_{};
    std::string path_str_{"./"};
    bool intrinsic_matrix_saved_ = false;

    std::chrono::system_clock::time_point time_start_ = std::chrono::system_clock::now();
    std::chrono::system_clock::time_point time_recv_ = std::chrono::system_clock::now();
    int recv_cnt_ = 0;
    
#ifdef HAS_OPENCV
    cv::Mat imgRGB_{};
    cv::Mat imgDepth_{};
#endif
};


int main(int argc, char** argv)
{
    std::string base_path_string = argc > 1 ? std::string{argv[1]} : ".";
    Record3DDemoApp app(base_path_string);
    auto exit = [](int) {
        is_quit.store(true, std::memory_order_release);
    };
    ::signal(SIGINT  , exit);
    ::signal(SIGABRT , exit);
    ::signal(SIGSEGV , exit);
    ::signal(SIGTERM , exit);

    app.Run();
}

#pragma clang diagnostic pop
