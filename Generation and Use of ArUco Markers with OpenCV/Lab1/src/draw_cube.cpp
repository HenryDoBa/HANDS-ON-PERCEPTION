#include <opencv2/opencv.hpp>
#include <opencv2/aruco.hpp>
#include <iostream>
#include <vector>
#include <string>
#include <map>

int main(int argc, char** argv) {
    
    std::string dictName = argv[1];
    int targetId = std::stoi(argv[2]);
    float markerLength = std::stof(argv[3]);
    std::string calibFile = argv[4];

    //Load Camera Calibration Data
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

    while (inputVideo.grab()) {
        cv::Mat image, imageCopy;
        inputVideo.retrieve(image);
        image.copyTo(imageCopy);

        std::vector<int> ids;
        std::vector<std::vector<cv::Point2f>> corners;
        
        // Detect markers in the current frame
        cv::aruco::detectMarkers(image, dictionary, corners, ids, params);

        if (ids.size() > 0) {
            std::vector<cv::Vec3d> rvecs, tvecs;

            // Estimate pose (rotation and translation) for each detected marker
            cv::aruco::estimatePoseSingleMarkers(corners, markerLength, camMatrix, distCoeffs, rvecs, tvecs);

            for (size_t i = 0; i < ids.size(); i++) {
                if (ids[i] == targetId) {

                    // Define the 8 vertices of a cube in 3D space
                    // Marker lies on the plane z=0 with center at (0,0,0)
                    // The cube "grows" upwards along negative Z (toward the camera)
                    float half = markerLength / 2.0f;
                    std::vector<cv::Point3f> cubePoints;

                    // Bottom face (on the marker)
                    cubePoints.push_back(cv::Point3f(-half, -half, 0));
                    cubePoints.push_back(cv::Point3f(half, -half, 0));
                    cubePoints.push_back(cv::Point3f(half, half, 0));
                    cubePoints.push_back(cv::Point3f(-half, half, 0));

                    // Top face (raised by markerLength)
                    cubePoints.push_back(cv::Point3f(-half, -half, markerLength));
                    cubePoints.push_back(cv::Point3f(half, -half, markerLength));
                    cubePoints.push_back(cv::Point3f(half, half, markerLength));
                    cubePoints.push_back(cv::Point3f(-half, half, markerLength));

                    // Project 3D points to 2D image coordinates
                    std::vector<cv::Point2f> imgPoints;
                    cv::projectPoints(cubePoints, rvecs[i], tvecs[i], camMatrix, distCoeffs, imgPoints);

                    // Draw bottom face edges (0-1-2-3)
                    for (int j = 0; j < 4; j++)
                        cv::line(imageCopy, imgPoints[j], imgPoints[(j + 1) % 4], cv::Scalar(255, 0, 0), 2);

                    // Draw top face edges (4-5-6-7)
                    for (int j = 4; j < 8; j++)
                        cv::line(imageCopy, imgPoints[j], imgPoints[4 + (j + 1) % 4], cv::Scalar(0, 255, 0), 2);

                    // Draw vertical edges connecting bottom and top faces
                    for (int j = 0; j < 4; j++)
                        cv::line(imageCopy, imgPoints[j], imgPoints[j + 4], cv::Scalar(0, 0, 255), 2);
                    
                    // Display label text near the cube
                    cv::putText(imageCopy, "Object: Cube AR", imgPoints[4], 
                                cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 1);
                }
            }
        }

        // Show the augmented reality image
        cv::imshow("Augmented Reality - Cube", imageCopy);
        if (cv::waitKey(1) == 27) break;
    }
    return 0;
}
