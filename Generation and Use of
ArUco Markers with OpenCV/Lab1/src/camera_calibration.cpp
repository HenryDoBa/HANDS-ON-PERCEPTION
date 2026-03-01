#include <opencv2/opencv.hpp>
#include <opencv2/aruco.hpp>
#include <iostream>
#include <vector>
#include <string>
#include <map>

int main(int argc, char** argv) {
    if (argc < 7) {
        std::cerr << "Usage: " << argv[0] 
                  << " <dict_name> <cols> <rows> <marker_m> <sep_m> <output.yml>" << std::endl;
        return -1;
    }

    // Parse command line arguments
    std::string dictName = argv[1];
    int cols = std::stoi(argv[2]);
    int rows = std::stoi(argv[3]);
    float markerLength = std::stof(argv[4]);
    float markerSep = std::stof(argv[5]);
    std::string outputFile = argv[6];

    // Dictionary mapping
    std::map<std::string, int> dictMap = {
        {"DICT_4X4_50", 0}, {"DICT_4X4_100", 1}, {"DICT_4X4_250", 2}, {"DICT_4X4_1000", 3},
        {"DICT_5X5_50", 4}, {"DICT_5X5_100", 5}, {"DICT_5X5_250", 6}, {"DICT_5X5_1000", 7},
        {"DICT_6X6_50", 8}, {"DICT_6X6_100", 9}, {"DICT_6X6_250", 10}, {"DICT_6X6_1000", 11},
        {"DICT_ARUCO_ORIGINAL", 16}
    };

    if (dictMap.find(dictName) == dictMap.end()) {
        std::cerr << "Invalid dictionary name!" << std::endl;
        return -1;
    }

    cv::Ptr<cv::aruco::Dictionary> dictionary = cv::aruco::getPredefinedDictionary(dictMap[dictName]);
    cv::Ptr<cv::aruco::GridBoard> board = cv::aruco::GridBoard::create(cols, rows, markerLength, markerSep, dictionary);

    cv::VideoCapture inputVideo(0);
    if (!inputVideo.isOpened()) {
        std::cerr << "Error: Could not open camera." << std::endl;
        return -1;
    }

    // Vectors lưu dữ liệu calibration
    std::vector<std::vector<cv::Point2f>> allCorners; // mỗi marker là 1 vector<Point2f>
    std::vector<int> allIds;                           // id tương ứng với mỗi marker
    std::vector<int> markerCounterPerFrame;           // số marker mỗi frame
    cv::Size imgSize;

    std::cout << "Press 'c' to capture, 's' to save, 'q' to quit." << std::endl;

    while (true) {
        cv::Mat image, imageCopy;
        inputVideo >> image;
        if (image.empty()) break;
        image.copyTo(imageCopy);
        imgSize = image.size();

        std::vector<int> ids;
        std::vector<std::vector<cv::Point2f>> corners;
        cv::aruco::detectMarkers(image, dictionary, corners, ids);

        if (!ids.empty())
            cv::aruco::drawDetectedMarkers(imageCopy, corners, ids);

        cv::imshow("Calibration", imageCopy);
        char key = (char)cv::waitKey(10);

        if (key == 'c' && !ids.empty()) {
            // Lưu từng marker vào allCorners và allIds
            for (size_t i = 0; i < corners.size(); i++) {
                allCorners.push_back(corners[i]);
                allIds.push_back(ids[i]);
            }
            markerCounterPerFrame.push_back((int)ids.size());
            std::cout << "Captured frame " << markerCounterPerFrame.size() << std::endl;
        }
        else if (key == 's') {
            if (markerCounterPerFrame.size() < 15) {
                std::cout << "Need at least 15 frames for calibration." << std::endl;
                continue;
            }

            cv::Mat camMatrix, distCoeffs;
            std::vector<cv::Mat> rvecs, tvecs;

            double repError = cv::aruco::calibrateCameraAruco(
                allCorners,
                allIds,
                markerCounterPerFrame,
                board,
                imgSize,
                camMatrix,
                distCoeffs,
                rvecs,
                tvecs
            );

            cv::FileStorage fs(outputFile, cv::FileStorage::WRITE);
            fs << "camera_matrix" << camMatrix;
            fs << "distortion_coefficients" << distCoeffs;
            fs << "reprojection_error" << repError;
            fs.release();

            std::cout << "Calibration saved to " << outputFile << " with error: " << repError << std::endl;
            break;
        }
        else if (key == 'q') {
            break;
        }
    }

    return 0;
}
