#include <opencv2/opencv.hpp>
#include <dirent.h>
#include <stdexcept>
#include <cstring>

#include "QMULset.h"

#define QMUL_TILT_COUNT 7
#define QMUL_PAN_COUNT 19
#define QMUL_IMG_SIZE 100

using namespace cv;
using namespace std;


/*  The class who will contain the set of images of the QMUL SET
*/

//return the image by knowing directly the vector of vector indices
Mat QMULset::getByIndex(int face, int tilt, int pan){
    assert (face >= 0 && face < img.size() && tilt >= 0 && tilt < QMUL_TILT_COUNT && pan >= 0 && pan < QMUL_PAN_COUNT);
    Mat toR = img[face][tilt * QMUL_PAN_COUNT + pan];
    return toR;
}

/*constructor take as param the path to the folder cotain the folders of subjects

*/
QMULset::QMULset(string path, bool UNIXenv){
    DIR* Qdir = opendir(path.c_str()); //load the main folder
    if (!Qdir) {
        throw invalid_argument("path does not exist");
    }
    struct dirent * dp;
    int i=0;
    img = vector<vector<Mat>>();
    while ((dp = readdir(Qdir)) != NULL) { //iterate over all model folders
        if (dp->d_name[0] == '.') {
            // Is .. or might be a system file like .DS_Store
            continue;
        }

        string foldpath;
        if (UNIXenv) {
           foldpath = path + "/" + dp->d_name; //construct path to current model if system is unix based
        } else {
            foldpath = path + "\\" + dp->d_name; //construct path to current model if system is DOS based
        }

        DIR* Fdir = opendir(foldpath.c_str()); //open model folder

        string name = string(dp->d_name);
        nameMap[name] = i; //insert the subject name in the map and map it to it's index (i)

        vector<Mat> poses; // Create the pose vector
        poses.resize(QMUL_TILT_COUNT * QMUL_PAN_COUNT); // ensure we have enough room for all poses

        struct dirent * fp;
        while ((fp = readdir(Fdir)) != NULL) { //iterate over all image and add them to the previously created vector
            if (fp->d_name[0] == '.') {
                // Is .. or might be a system file like .DS_Store
                continue;
            }
            Mat pose = imread(foldpath + (UNIXenv ? "/" : "\\") + fp->d_name);
            cvtColor(pose, pose, CV_BGR2GRAY);

            char* poseName = strtok(fp->d_name, "_");
            char* tiltCode = strtok(NULL, "_");
            char* panCode = strtok(NULL, "_");
            // Remove .ras extension
            panCode[strlen(panCode) - 4] = '\0';
            // Code is 60 to 120 for tilt converted to 0 to 6
            int tilt = stoi(string(tiltCode)) / 10 - 6;
            // Code is 0 to 180 for pan converted to 0 to 18
            int pan = stoi(string(panCode)) / 10;
            poses[tilt * QMUL_PAN_COUNT + pan] = pose;
        }
        img.push_back(poses); //push a new vector of image in our vector of model
        i++;
    }
}

//get image by knoing subject name, tilt in [-30;30] and pan in [-90; 90]
Mat QMULset::get(string subjectName, int tilt, int pan) {
    map<string, int>::iterator pair = nameMap.find(subjectName);
    if (pair == nameMap.end()) {
        throw invalid_argument("Not a valid subject name: " + subjectName);
    }
    return get(pair->second, tilt, pan); //map name to index and pass to another getter
}

//get image by knoing subject index, tilt in [-30;30] and pan in [-90; 90]
Mat QMULset::get(int subjectNameIndex, int tilt, int pan){
    return getByIndex(subjectNameIndex, tilt/10 + 3, pan/10 + 9); //transform deegre into index
}

//get all the image of a given subject name
void QMULset::getPersonSet(String subjectName, vector<Mat>& set) {
    map<string, int>::iterator pair = nameMap.find(subjectName);
    if (pair == nameMap.end()) {
        throw invalid_argument("Not a valid subject name: " + subjectName);
    }
    getPersonSet(pair->second, set); //translat and pass
}

//get all the image of a given subject index
void QMULset::getPersonSet(int index, vector<Mat>& set){
    assert (index >= 0 && index < img.size());
    set = img[index]; //return the vector of images correspondong to the index
}

//Return all images corresponding to a given (tilt,pan)
void QMULset::getPoseSet(int tilt, int pan, vector<Mat>& set) {
    for (int i = 1; i<img.size(); i++) {
        set.push_back(get(i,tilt,pan));
    }
}

size_t getClosestCoarse(vector<int> classes, int value) {
    int minDistance = 0xFFFF;
    size_t closest = 0;
    for (size_t i = 0; i < classes.size(); i++) {
        int coarse = classes[i];
        int diff = value - coarse;
        int distance = diff < 0 ? -diff : diff;
        if (distance <= minDistance) {
            minDistance = distance;
            closest = i;
        }
    }
    return closest;
}

void QMULset::getCoarsePoseSet(vector<int> tiltClasses, vector<int> panClasses, int tiltIndex, int panIndex, vector<Mat>& set) {
    vector<int> closeTilts, closePans;
    // Find all the tilts closest to the coarse one
    for (size_t i = 0; i < QMUL_TILT_COUNT; i++) {
        int tilt = i * 10 - 30;
        if (getClosestCoarse(tiltClasses, tilt) == tiltIndex) {
            closeTilts.push_back(tilt);
        }
    }
    // Find all the pans closest to the coarse one
    for (size_t i = 0; i < QMUL_PAN_COUNT; i++) {
        int pan = i * 10 - 90;
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

//get one big vector with all images
void QMULset::getAll(vector<Mat>& set) {
    for (int i =1; i<img.size(); i++) { //just flatten the vector of vector
        set.insert(end(set), begin(img[i]), end(img[i]));
    }
}


Mat QMULset::getAllImage(string subjectName) {
    map<string, int>::iterator pair = nameMap.find(subjectName);
    if (pair == nameMap.end()) {
        throw invalid_argument("Not a valid subject name: " + subjectName);
    }
    return getAllImage(pair->second);
}

//return a big image containing all other image
Mat QMULset::getAllImage(int subjectIndex) {
    // Create an image of that can contain all images, ordered by tilt and pan
    Mat all = Mat::zeros(QMUL_TILT_COUNT * QMUL_IMG_SIZE, QMUL_PAN_COUNT * QMUL_IMG_SIZE, CV_8U);
    // For each image in the set
    for (size_t tilt = 0; tilt < QMUL_TILT_COUNT; tilt++) {
        for (size_t pan = 0; pan < QMUL_PAN_COUNT; pan++) {
            // Get the slot of the full image that will be used
            size_t allRow = tilt * QMUL_IMG_SIZE;
            size_t allCol = pan * QMUL_IMG_SIZE;
            Mat slot = all(Range(allRow, allRow + QMUL_IMG_SIZE), Range(allCol, allCol + QMUL_IMG_SIZE));
            // Copy the pose into the slot
            Mat pose = getByIndex(subjectIndex, tilt, pan);
            pose.copyTo(slot);
        }
    }
    return all;
}
