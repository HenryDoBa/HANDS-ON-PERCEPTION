#include <opencv2/opencv.hpp>
#include <opencv2/aruco.hpp>
#include <iostream>
#include <string>
#include <map>

int main(int argc, char** argv) {
    std::string dictName = argv[1];
    int markerId = std::stoi(argv[2]);
    int markerSize = std::stoi(argv[3]);
    std::string fileName = argv[4];

    // Map string names to OpenCV Dictionary enums
    std::map<std::string, int> dictMap = {
        {"DICT_4X4_50", 0}, {"DICT_4X4_100", 1}, {"DICT_4X4_250", 2}, {"DICT_4X4_1000", 3},
        {"DICT_5X5_50", 4}, {"DICT_5X5_100", 5}, {"DICT_5X5_250", 6}, {"DICT_5X5_1000", 7},
        {"DICT_6X6_50", 8}, {"DICT_6X6_100", 9}, {"DICT_6X6_250", 10}, {"DICT_6X6_1000", 11},
        {"DICT_7X7_50", 12}, {"DICT_7X7_100", 13}, {"DICT_7X7_250", 14}, {"DICT_7X7_1000", 15},
        {"DICT_ARUCO_ORIGINAL", 16}
    };

    if (dictMap.find(dictName) == dictMap.end()) {
        std::cerr << "Error: Dictionary " << dictName << " not found!" << std::endl;
        return -1;
    }

    cv::Ptr<cv::aruco::Dictionary> dictionary = cv::aruco::getPredefinedDictionary(dictMap[dictName]);

    cv::Mat markerImage;
    cv::aruco::drawMarker(dictionary, markerId, markerSize, markerImage, 1);

    // Add white border (Quiet Zone)
    int borderSize = markerSize / 10;
    cv::Mat markerWithBorder;
    cv::copyMakeBorder(markerImage, markerWithBorder, borderSize, borderSize, borderSize, borderSize, 
                       cv::BORDER_CONSTANT, cv::Scalar(255));

    if (cv::imwrite(fileName, markerWithBorder)) {
        std::cout << "Successfully generated " << dictName << " marker ID " << markerId << " to " << fileName << std::endl;
    }

    return 0;
}