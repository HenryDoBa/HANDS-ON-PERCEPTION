#include <opencv2/opencv.hpp>
#include <opencv2/aruco.hpp>
#include <iostream>

int main(int argc, char** argv) {
    if (argc < 5) {
        std::cerr << "Usage: " << argv[0] << " <dict_name> <marker_id> <marker_len_m> <calib.yml>" << std::endl;
        return -1;
    }

    // Load camera calibration data from file
    cv::Mat camMatrix, distCoeffs;
    cv::FileStorage fs(argv[4], cv::FileStorage::READ);
    fs["camera_matrix"] >> camMatrix;
    fs["distortion_coefficients"] >> distCoeffs;

    // Load ArUco dictionary (currently using original dictionary)
    cv::Ptr<cv::aruco::Dictionary> dict = cv::aruco::getPredefinedDictionary(cv::aruco::DICT_ARUCO_ORIGINAL);

    // Open default camera
    cv::VideoCapture cap(0);

    while (cap.grab()) {
        cv::Mat img, imgCopy;
        cap.retrieve(img);
        img.copyTo(imgCopy);

        std::vector<int> ids;
        std::vector<std::vector<cv::Point2f>> corners;

        // Detect markers in the image
        cv::aruco::detectMarkers(img, dict, corners, ids);

        if (ids.size() > 0) {
            std::vector<cv::Vec3d> rvecs, tvecs;

            // Estimate pose of each detected marker
            cv::aruco::estimatePoseSingleMarkers(corners, std::stof(argv[3]), camMatrix, distCoeffs, rvecs, tvecs);

            for (int i = 0; i < ids.size(); i++) {
                if (ids[i] == std::stoi(argv[2])) {
                    // Draw coordinate axes on the marker
                    cv::drawFrameAxes(imgCopy, camMatrix, distCoeffs, rvecs[i], tvecs[i], 0.1);
                    
                    // Display X, Y, Z translation values
                    std::string text = "X: " + std::to_string(tvecs[i][0]) +
                                       " Y: " + std::to_string(tvecs[i][1]) +
                                       " Z: " + std::to_string(tvecs[i][2]);
                    cv::putText(imgCopy, text, cv::Point(10, 50), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 255, 0), 2);
                }
            }
        }

        // Show the result image with axes and text
        cv::imshow("Pose Estimation", imgCopy);

        // Exit if 'q' is pressed
        if (cv::waitKey(30) == 'q') break;
    }

    return 0;
}
