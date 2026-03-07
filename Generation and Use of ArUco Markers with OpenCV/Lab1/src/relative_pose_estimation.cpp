#include <opencv2/opencv.hpp>
#include <opencv2/aruco.hpp>
#include <iostream>
#include <string>
#include <map>

int main(int argc, char** argv) {
    if (argc < 6) {
        std::cerr << "Usage: " << argv[0] << " <dict_name> <id1> <id2> <marker_len_m> <calib.yml>" << std::endl;
        return -1;
    }

    std::string dictName = argv[1];
    int id1 = std::stoi(argv[2]);          
    int id2 = std::stoi(argv[3]);          
    float markerLength = std::stof(argv[4]); 
    std::string calibFile = argv[5];       

    //Load camera calibration parameters
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

    if (dictMap.find(dictName) == dictMap.end()) {
        std::cerr << "Error: Dictionary " << dictName << " not found!" << std::endl;
        return -1;
    }

    cv::Ptr<cv::aruco::Dictionary> dictionary = cv::aruco::getPredefinedDictionary(dictMap[dictName]);
    cv::Ptr<cv::aruco::DetectorParameters> params = cv::aruco::DetectorParameters::create();

    cv::VideoCapture inputVideo(0);
    if (!inputVideo.isOpened()) {
        std::cerr << "Could not open camera." << std::endl;
        return -1;
    }

    while (inputVideo.grab()) {
        cv::Mat image, imageCopy;
        inputVideo.retrieve(image);
        image.copyTo(imageCopy);

        std::vector<int> ids;
        std::vector<std::vector<cv::Point2f>> corners;

        cv::aruco::detectMarkers(image, dictionary, corners, ids, params);

        if (ids.size() >= 2) {
            std::vector<cv::Vec3d> rvecs, tvecs;
            cv::aruco::estimatePoseSingleMarkers(corners, markerLength, camMatrix, distCoeffs, rvecs, tvecs);

            int idx1 = -1, idx2 = -1;
            for (size_t i = 0; i < ids.size(); i++) {
                if (ids[i] == id1) idx1 = i;
                if (ids[i] == id2) idx2 = i;
            }

            // Only proceed if both requested markers are visible
            if (idx1 != -1 && idx2 != -1) {
                
                // Draw green frame, red square, and ID for both markers
                std::vector<int> targetIds = {ids[idx1], ids[idx2]};
                std::vector<std::vector<cv::Point2f>> targetCorners = {corners[idx1], corners[idx2]};
                cv::aruco::drawDetectedMarkers(imageCopy, targetCorners, targetIds);

                // Draw coordinate axes for both markers
                cv::drawFrameAxes(imageCopy, camMatrix, distCoeffs, rvecs[idx1], tvecs[idx1], markerLength);
                cv::drawFrameAxes(imageCopy, camMatrix, distCoeffs, rvecs[idx2], tvecs[idx2], markerLength);

                // Determine which marker is left-most and right-most based on their X translation
                // In the camera frame, negative X is left, positive X is right
                int idxL = (tvecs[idx1][0] < tvecs[idx2][0]) ? idx1 : idx2;
                int idxR = (tvecs[idx1][0] < tvecs[idx2][0]) ? idx2 : idx1;

                cv::Mat R_L;
                // Convert left marker's rotation vector into a 3x3 rotation matrix
                cv::Rodrigues(rvecs[idxL], R_L); 

                // Create column matrices for translations
                cv::Mat tL = (cv::Mat_<double>(3, 1) << tvecs[idxL][0], tvecs[idxL][1], tvecs[idxL][2]);
                cv::Mat tR = (cv::Mat_<double>(3, 1) << tvecs[idxR][0], tvecs[idxR][1], tvecs[idxR][2]);

                // Calculate relative position: t_rel = R_L^T * (t_R - t_L)
                cv::Mat t_rel = R_L.t() * (tR - tL);

                double x_rel = t_rel.at<double>(0, 0);
                double y_rel = t_rel.at<double>(1, 0);
                double z_rel = t_rel.at<double>(2, 0);

                // Overlay relative coordinates on the video
                std::string coords = "Rel X: " + std::to_string(x_rel).substr(0, 6) + " m   " +
                                     "Rel Y: " + std::to_string(y_rel).substr(0, 6) + " m   " +
                                     "Rel Z: " + std::to_string(z_rel).substr(0, 6) + " m";
                
                // Display at the top-left corner of the screen
                cv::putText(imageCopy, coords, cv::Point(20, 40), 
                            cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 0), 2);
            }
        }

        cv::imshow("Relative Pose Estimation", imageCopy);
        if (cv::waitKey(1) == 27) break; 
    }

    return 0;
}