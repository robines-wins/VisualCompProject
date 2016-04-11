#include <opencv2/opencv.hpp>
#include <dirent.h>
#include <stdexcept>
#include <algorithm>
#include <cstring>
#include <cstdio>

#include "HPset.h"

#define HP_ID_COUNT 15
#define HP_SERIE_COUNT 2
#define HP_TILT_COUNT 5
#define HP_PAN_COUNT 13
#define HP_IMG_SIZE 100

using namespace cv;
using namespace std;

Mat crop(Mat image, int centerX, int centerY) {
    Mat cropped = Mat::zeros(100, 100, CV_8U);
    Mat face = image(Range(max(centerY - 50, 0), min(centerY + 50, image.rows)), Range(max(centerX - 50, 0), min(centerX + 50, image.cols)));
    for (int row = 0; row < face.rows; row++) {
        for (int col = 0; col < face.cols; col++) {
            cropped.at<uchar>(row, col) = face.at<uchar>(row, col);
        }
    }
    assert (cropped.rows == 100 && cropped.cols == 100);
    return cropped;
}

size_t faceIndex(int id, int serie) {
    id -= 1;
    serie -= 1;
    assert (id >= 0 && id < 15 && serie >= 0 && serie < 2);
    return id * HP_SERIE_COUNT + serie;
}

size_t poseIndex(int tilt, int pan) {
    tilt += 30;
    tilt /= 15;
    pan += 90;
    pan /= 15;
    assert (tilt >= 0 && tilt < HP_TILT_COUNT && pan >= 0 && pan < HP_PAN_COUNT);
    return tilt * HP_PAN_COUNT + pan;
}

/*
  constructor take as param the path to the folder cotain the folders of subjects
*/
HPset::HPset(string path, bool UNIXenv) {
    DIR* mainDir = opendir(path.c_str());
    if (!mainDir) {
        throw invalid_argument("path does not exist");
    }
    // Prepare image storage
    images.resize(HP_ID_COUNT * HP_SERIE_COUNT);
    for (size_t i = 0; i < images.size(); i++) {
        images[i].resize(HP_TILT_COUNT * HP_PAN_COUNT);
    }
    // Read the images for each person
    struct dirent* personDir;
    while ((personDir = readdir(mainDir))) {
        char* dirName = personDir->d_name;
        // Exclude files that aren't person folders
        if (strlen(dirName) != 8 || strncmp(dirName, "Person", 6) != 0) {
            continue;
        }
        // Get the path containing the person's files
        string personDirPath = path + (UNIXenv ? "/" : "\\") + string(dirName);
        DIR* dir = opendir(personDirPath.c_str());
        // Read the files for the person
        struct dirent* personFile;
        while ((personFile = readdir(dir))) {
            char* fileName = personFile->d_name;
            // Exclude files that aren't person files
            size_t length = strlen(fileName);
            if (length < 6 || strncmp(fileName, "person", 6) != 0) {
                continue;
            }
            // Only scan .txt files
            char* extension = fileName + length - 4;
            if (strncmp(extension, ".txt", 4) != 0) {
                continue;
            }
            // Read the info file
            string filePath = personDirPath + (UNIXenv ? "/" : "\\") + string(fileName);
            FILE* info = fopen(filePath.c_str(), "r");
            char imageName[256];
            int id, serie, number, tilt, pan, centerX, centerY;
            fscanf(info, "%s\n\nFace\n%d\n%d", imageName, &centerX, &centerY);
            sscanf(imageName, "person%2d%1d%2d%d%d.jpg", &id, &serie, &number, &tilt, &pan);
            fclose(info);
            // Discard images with tilt angles greater than +- 30
            if (tilt < -30 || tilt > 30) {
                continue;
            }
            // Read the image file, convert to gray and crop the face
            Mat image = imread(personDirPath + (UNIXenv ? "/" : "\\") + string(imageName));
            cvtColor(image, image, CV_BGR2GRAY);
            image = crop(image, centerX, centerY);
            // Add image to set
            images[faceIndex(id, serie)][poseIndex(tilt, pan)] = image;
        }
    }
}

Mat HPset::get(int personId, int serie, int tilt, int pan) {
    return images[faceIndex(personId, serie)][poseIndex(tilt, pan)];
}

void HPset::getPersonSet(int personId, int serie, vector<Mat>& set) {
    set = images[faceIndex(personId, serie)];
}

void HPset::getPoseSet(int tilt, int pan, vector<Mat>& set) {
    size_t pose = poseIndex(tilt, pan);
    for (size_t index = 0; index < HP_ID_COUNT * HP_SERIE_COUNT; index++) {
        set.push_back(images[index][pose]);
    }
}

void HPset::getCoarsePoseSet(vector<int> tiltClasses, vector<int> panClasses, int tiltIndex, int panIndex, vector<Mat>& set) {
    vector<int> closeTilts, closePans;
    // Find all the tilts closest to the coarse one
    for (size_t i = 0; i < HP_TILT_COUNT; i++) {
        int tilt = i * 15 - 30;
        if (getClosestCoarse(tiltClasses, tilt) == tiltIndex) {
            closeTilts.push_back(tilt);
        }
    }
    // Find all the pans closest to the coarse one
    for (size_t i = 0; i < HP_PAN_COUNT; i++) {
        int pan = i * 15 - 90;
        if (getClosestCoarse(panClasses, pan) == panIndex) {
            closePans.push_back(pan);
        }
    }
    // Get all poses for the found close tilt and close pan combinations
    for (size_t i = 0; i < closeTilts.size(); i++) {
        for (size_t j = 0; j < closePans.size(); j++) {
            getPoseSet(closeTilts[i], closePans[j], set);
        }
    }
}

Mat HPset::getAllImage(int personId, int serie) {
    // Create an image of that can contain all images, ordered by tilt and pan
    Mat all = Mat::zeros(HP_TILT_COUNT * HP_IMG_SIZE, HP_PAN_COUNT * HP_IMG_SIZE, CV_8U);
    // For each image in the set
    size_t face = faceIndex(personId, serie);
    for (size_t tilt = 0; tilt < HP_TILT_COUNT; tilt++) {
        for (size_t pan = 0; pan < HP_PAN_COUNT; pan++) {
            // Get the slot of the full image that will be used
            size_t allRow = tilt * HP_IMG_SIZE;
            size_t allCol = pan * HP_IMG_SIZE;
            Mat slot = all(Range(allRow, allRow + HP_IMG_SIZE), Range(allCol, allCol + HP_IMG_SIZE));
            // Copy the pose into the slot
            Mat pose = images[face][tilt * HP_PAN_COUNT + pan];
            pose.copyTo(slot);
        }
    }
    return all;
}
