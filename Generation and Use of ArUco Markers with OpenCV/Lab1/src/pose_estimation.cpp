#include <opencv2/opencv.hpp>
#include <opencv2/aruco.hpp>
#include <iostream>
#include <string>
#include <map>

int main(int argc, char** argv) {

    std::string dictName = argv[1];
    int targetId = std::stoi(argv[2]);
    float markerLength = std::stof(argv[3]);
    std::string calibFile = argv[4];

    //Load camera calibration parameters from the .yml file
    cv::Mat camMatrix, distCoeffs;
    cv::FileStorage fs(calibFile, cv::FileStorage::READ);
    if (!fs.isOpened()) {
        std::cerr << "Could not open calibration file: " << calibFile << std::endl;
        return -1;
    }
    fs["camera_matrix"] >> camMatrix;
    fs["distortion_coefficients"] >> distCoeffs;
    fs.release();

    std::map<std::string, int> dictMap = {
        {"DICT_4X4_50", 0}, {"DICT_4X4_100", 1}, {"DICT_4X4_250", 2}, {"DICT_4X4_1000", 3},
        {"DICT_5X5_50", 4}, {"DICT_5X5_100", 5}, {"DICT_5X5_250", 6}, {"DICT_5X5_1000", 7},
        {"DICT_6X6_50", 8}, {"DICT_6X6_100", 9}, {"DICT_6X6_250", 10}, {"DICT_6X6_1000", 11},
        {"DICT_ARUCO_ORIGINAL", 16}
    };

    cv::Ptr<cv::aruco::Dictionary> dictionary = cv::aruco::getPredefinedDictionary(dictMap[dictName]);
    cv::Ptr<cv::aruco::DetectorParameters> params = cv::aruco::DetectorParameters::create();

    cv::VideoCapture inputVideo(0);
    if (!inputVideo.isOpened()) {
        std::cerr << "Could not open camera." << std::endl;
        return -1;
    }

    std::cout << "Starting Pose Estimation for ID: " << targetId << ". Press ESC to quit." << std::endl;

    while (inputVideo.grab()) {
        cv::Mat image, imageCopy;
        inputVideo.retrieve(image);
        image.copyTo(imageCopy);

        std::vector<int> ids;
        std::vector<std::vector<cv::Point2f>> corners;

        // Detect markers in the current frame
        cv::aruco::detectMarkers(image, dictionary, corners, ids, params);

        if (ids.size() > 0) {
            // Pose Estimation: returns rotation vectors (rvecs) and translation vectors (tvecs)
            std::vector<cv::Vec3d> rvecs, tvecs;
            cv::aruco::estimatePoseSingleMarkers(corners, markerLength, camMatrix, distCoeffs, rvecs, tvecs);

            for (size_t i = 0; i < ids.size(); i++) {
                // Only process/draw if it matches the target ID requested from the command line
                if (ids[i] == targetId) {
                    // Draw basic bounding box and ID
                    cv::aruco::drawDetectedMarkers(imageCopy, corners, ids);

                    // Draw 3D coordinate axes (Red: X, Green: Y, Blue: Z)
                    // Axis length is set to 0.1m (10cm)
                    cv::drawFrameAxes(imageCopy, camMatrix, distCoeffs, rvecs[i], tvecs[i], 0.1);

                    // Extract X, Y, Z coordinates from the translation vector (tvecs)
                    double x = tvecs[i][0];
                    double y = tvecs[i][1];
                    double z = tvecs[i][2];

                    // Create text string to display coordinates
                    std::string coords = "X: " + std::to_string(x).substr(0, 5) + 
                                         " Y: " + std::to_string(y).substr(0, 5) + 
                                         " Z: " + std::to_string(z).substr(0, 5);

                    // Overlay coordinates onto the video (positioned near the marker corner)
                    cv::putText(imageCopy, coords, corners[i][0], 
                                cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 255, 0), 2);
                    
                    // Log to terminal for debugging
                    std::cout << "Marker ID " << targetId << " Position -> " << coords << "\r" << std::flush;
                }
            }
        }

        cv::imshow("Pose Estimation", imageCopy);
        if (cv::waitKey(1) == 27) break;
    }

    return 0;
}