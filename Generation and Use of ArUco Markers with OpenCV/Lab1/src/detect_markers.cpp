#include <opencv2/opencv.hpp>
#include <opencv2/aruco.hpp>
#include <iostream>
#include <vector>
#include <string>
#include <map>

int main(int argc, char** argv) {
    std::string dictName = argv[1];
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
    cv::Ptr<cv::aruco::DetectorParameters> parameters = cv::aruco::DetectorParameters::create();

    cv::VideoCapture inputVideo(0);
    if (!inputVideo.isOpened()) {
        std::cerr << "Error: Could not open camera." << std::endl;
        return -1;
    }

    std::cout << "Starting detection. Press ESC to exit." << std::endl;

    while (true) {
        cv::Mat image, imageCopy;
        inputVideo >> image; 
        if (image.empty()) break;

        image.copyTo(imageCopy);

        std::vector<int> ids;
        std::vector<std::vector<cv::Point2f>> corners;

        //Detect Markers
        cv::aruco::detectMarkers(image, dictionary, corners, ids, parameters);

        //Draw information on screen if markers are detected
        if (ids.size() > 0) {
            // Draw default green boundaries and IDs
            cv::aruco::drawDetectedMarkers(imageCopy, corners, ids);

            // Draw a small red square at the top-left corner (Corner index 0) for each marker
            for (size_t i = 0; i < corners.size(); i++) {
                // Get the coordinates of the first corner (top-left)
                cv::Point2f topLeft = corners[i][0];
                
                // Define a small red square (10x10) around that point
                cv::Rect redSquare(topLeft.x - 5, topLeft.y - 5, 10, 10);
                cv::rectangle(imageCopy, redSquare, cv::Scalar(0, 0, 255), -1); // -1 to fill the square
            }
        }

        cv::imshow("ArUco Detection", imageCopy);

        if (cv::waitKey(30) == 27) break;
    }

    return 0;
}