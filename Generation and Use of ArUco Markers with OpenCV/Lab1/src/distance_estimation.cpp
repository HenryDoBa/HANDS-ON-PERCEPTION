#include <opencv2/opencv.hpp>
#include <opencv2/aruco.hpp>
#include <iostream>
#include <vector>
#include <string>
#include <map>
#include <cmath>

int main(int argc, char** argv) {
    std::string dictName = argv[1];
    int id1 = std::stoi(argv[2]);          // First target marker ID
    int id2 = std::stoi(argv[3]);          // Second target marker ID
    float markerLength = std::stof(argv[4]); // Marker side length in meters
    std::string calibFile = argv[5];       // Camera calibration file

    //Load camera calibration data
    cv::Mat camMatrix, distCoeffs;
    cv::FileStorage fs(calibFile, cv::FileStorage::READ);
    if (!fs.isOpened()) {
        std::cerr << "Could not open calibration file!" << std::endl;
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

    while (inputVideo.grab()) {
        cv::Mat image, imageCopy;
        inputVideo.retrieve(image);
        image.copyTo(imageCopy);

        std::vector<int> ids;
        std::vector<std::vector<cv::Point2f>> corners;

        // Detect markers in the current frame
        cv::aruco::detectMarkers(image, dictionary, corners, ids, params);

        if (ids.size() >= 2) {
            std::vector<cv::Vec3d> rvecs, tvecs;

            // Estimate pose (rotation and translation) for each detected marker
            cv::aruco::estimatePoseSingleMarkers(corners, markerLength, camMatrix, distCoeffs, rvecs, tvecs);

            int idx1 = -1, idx2 = -1;
            for (int i = 0; i < ids.size(); i++) {
                if (ids[i] == id1) idx1 = i;
                if (ids[i] == id2) idx2 = i;
            }

            // If both target markers are detected in the frame
            if (idx1 != -1 && idx2 != -1) {
                // Get translation vectors (tvecs) representing 3D positions of marker centers
                cv::Vec3d t1 = tvecs[idx1];
                cv::Vec3d t2 = tvecs[idx2];

                // Compute 3D Euclidean distance between the two markers
                double distance = std::sqrt(std::pow(t1[0] - t2[0], 2) +
                                            std::pow(t1[1] - t2[1], 2) +
                                            std::pow(t1[2] - t2[2], 2));

                // Compute 2D image centers of the markers to draw a connecting line
                cv::Point2f center1(0, 0), center2(0, 0);
                for (int j = 0; j < 4; j++) {
                    center1 += corners[idx1][j];
                    center2 += corners[idx2][j];
                }
                center1 *= (1.0 / 4.0);
                center2 *= (1.0 / 4.0);

                // Draw a line connecting the two marker centers
                cv::line(imageCopy, center1, center2, cv::Scalar(0, 255, 255), 3);

                // Display the 3D distance on the image
                std::string distText = "Distance: " + std::to_string(distance) + " m";
                cv::Point2f midPoint = (center1 + center2) * 0.5;
                cv::putText(imageCopy, distText, midPoint, cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 255), 2);
            }
        }

        // Show the frame with distance visualization
        cv::imshow("Distance Estimation", imageCopy);
        if (cv::waitKey(1) == 27) break;
    }
    return 0;
}
