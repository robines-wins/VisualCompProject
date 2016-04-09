#include <opencv2/opencv.hpp>
#include <dirent.h>
#include <stdexcept>

#include "QMULset.h"

using namespace cv;
using namespace std;


/*  The class who will contain the set of images of the QMUL SET
*/

//return the image by knowing directly the vector of vector indices
Mat QMULset::getByIndex(int face, int tilt, int pan){
    Mat toR = img[face][tilt*shiftPerTilt + pan];
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
    readdir(Qdir); //the 2 first item return by read dir are parent and current folder we don't want to iterate on them so we skip them
    readdir(Qdir);
    int i=0;
    img = vector<vector<Mat>>();
    while ((dp = readdir(Qdir)) != NULL) { //iterate over all model folders
        string foldpath;
        cout << dp->d_name << endl; //display the folder name, very optionnal

        if (UNIXenv) {
           foldpath = path + "/" + dp->d_name; //construct path to current model if system is unix based
        }
        else{
            foldpath = path + "\\" + dp->d_name; //construct path to current model if system is DOS based
        }

        DIR* Fdir = opendir(foldpath.c_str()); //open model folder

        string name = dp->d_name;
        name.erase(name.length()-4); //erase grey at the end of folder name to get subject name
        nameMap.insert(pair<string, int>(name, i)); //insert the subject name in the map and map it to it's index (i)
        img.push_back(vector<Mat>()); //push a new vector of image in our vector of model
        i++;

        struct dirent * fp;
        readdir(Fdir); //same as before, avoid ../ and ./
        readdir(Fdir);
        while ((fp = readdir(Fdir)) != NULL) { //iterate over all image and add them to the previously crated vector

            Mat yolo = imread(foldpath + "/" + fp->d_name);
            cvtColor(yolo, yolo, CV_BGR2GRAY);
            img.back().push_back(yolo.clone());
        }
    }
}

//get image by knoing subject name, tilt in [-30;30] and pan in [0;180]
Mat QMULset::get(string subjectName, int tilt, int pan){

    return get(nameMap.find(subjectName)->second, tilt, pan); //map name to index and pass to another getter
}

//get image by knoing subject index, tilt in [-30;30] and pan in [0;180]
Mat QMULset::get(int subjectNameIndex, int tilt, int pan){
    return getByIndex(subjectNameIndex, (tilt+30)/10, pan/10); //transform deegre into index
}

//get all the image of a given subject name
vector<Mat> QMULset::getSet(String subjectName){
    return getSet(nameMap.find(subjectName)->second); //translat and pass
}

//get all the image of a given subject index
vector<Mat> QMULset::getSet(int index){
    return img[index]; //return the vector of images correspondong to the index
}

//get one big vector with all images
vector<Mat> QMULset::getAll(){
    vector<Mat> all = img.front();
    for (int i =1; i<img.size(); i++) { //just flatten the vector of vector
        all.insert(end (all), begin(img[i]), end(img[i]));
    }
    return all;
}


//Return all images corresponding to a given (tilt,pan)
vector<Mat> QMULset::getSet(int tilt, int pan){
    vector<Mat> all;
    for (int i = 1; i<img.size(); i++) {
        all.push_back(get(i,tilt,pan));
    }
    return all;
}

//return a big image containing all other image TO BE FINISHED
Mat QMULset::allImageFromSubject(int subjectIndex){
    Mat canvas = Mat(img[subjectIndex].front().size(), img[subjectIndex].front().type());
    return canvas;
}
