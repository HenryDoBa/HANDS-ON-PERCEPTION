#include <opencv2/opencv.hpp>
#include <opencv2/aruco.hpp>
#include <iostream>
#include <string>
#include <map>

int main(int argc, char** argv) {

    std::string dictName = argv[3];
    int markersX = std::stoi(argv[1]);
    int markersY = std::stoi(argv[2]);
    int markerLength = std::stoi(argv[4]);
    int markerSeparation = std::stoi(argv[5]);
    std::string fileName = argv[6];

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

    // Get the dictionary
    cv::Ptr<cv::aruco::Dictionary> dictionary = cv::aruco::getPredefinedDictionary(dictMap[dictName]);

    // Create the GridBoard object
    cv::Ptr<cv::aruco::GridBoard> board = cv::aruco::GridBoard::create(
        markersX, markersY, (float)markerLength, (float)markerSeparation, dictionary);

    // Calculate total image size
    cv::Size imageSize;
    imageSize.width = markersX * (markerLength + markerSeparation) + markerSeparation;
    imageSize.height = markersY * (markerLength + markerSeparation) + markerSeparation;
    
    cv::Mat boardImage;
    board->draw(imageSize, boardImage, markerSeparation, 1);

    // Add white border (Quiet Zone) around the board
    cv::Mat boardWithBorder;
    cv::copyMakeBorder(boardImage, boardWithBorder, 40, 40, 40, 40, cv::BORDER_CONSTANT, cv::Scalar(255));

    if (cv::imwrite(fileName, boardWithBorder)) {
        std::cout << "Successfully generated " << dictName << " board to " << fileName << std::endl;
    }

    return 0;
}