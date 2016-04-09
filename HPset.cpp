#include <opencv2/opencv.hpp>
#include <dirent.h>
#include <stdexcept>
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

Mat crop(Mat& image, int centerX, int centerY) {
    Mat cropped = Mat::zeros(100, 100, CV_8U);
    Mat face = image(Range(centerY - 50, centerY + 50), Range(centerX - 50, centerX + 50));
    face.copyTo(cropped);
    return cropped;
}

size_t faceIndex(int id, int serie) {
    id -= 1;
    serie -= 1;
    assert(id >= 0 && id < 15 && serie >= 0 && serie < 2);
    return id * HP_SERIE_COUNT + serie;
}

size_t poseIndex(int tilt, int pan) {
    tilt += 30;
    tilt /= 15;
    pan += 90;
    pan /= 15;
    assert(tilt >= 0 && tilt < HP_TILT_COUNT && pan >= 0 && pan < HP_PAN_COUNT);
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
