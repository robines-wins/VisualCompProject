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
        name.erase(name.length()-4); //erase grey at the end of folder name to get subject name
        nameMap.insert(pair<string, int>(name, i)); //insert the subject name in the map and map it to it's index (i)

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

//get image by knoing subject name, tilt in [-30;30] and pan in [0;180]
Mat QMULset::get(string subjectName, int tilt, int pan){
    return get(nameMap.find(subjectName)->second, tilt, pan); //map name to index and pass to another getter
}

//get image by knoing subject index, tilt in [-30;30] and pan in [0;180]
Mat QMULset::get(int subjectNameIndex, int tilt, int pan){
    return getByIndex(subjectNameIndex, tilt/10 + 3, pan/10 + 9); //transform deegre into index
}

//get all the image of a given subject name
void QMULset::getPersonSet(String subjectName, vector<Mat>& set) {
    getPersonSet(nameMap.find(subjectName)->second, set); //translat and pass
}

//get all the image of a given subject index
void QMULset::getPersonSet(int index, vector<Mat>& set){
    set = img[index]; //return the vector of images correspondong to the index
}

//Return all images corresponding to a given (tilt,pan)
void QMULset::getPoseSet(int tilt, int pan, vector<Mat>& set) {
    for (int i = 1; i<img.size(); i++) {
        set.push_back(get(i,tilt,pan));
    }
}

//get one big vector with all images
void QMULset::getAll(vector<Mat>& set) {
    for (int i =1; i<img.size(); i++) { //just flatten the vector of vector
        set.insert(end(set), begin(img[i]), end(img[i]));
    }
}


Mat QMULset::getAllImage(string subjectName) {
    return getAllImage(nameMap.find(subjectName)->second);
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
