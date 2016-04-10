#include <cstdlib>
#include <iostream>
#include <opencv2/opencv.hpp>

#include "main.h"
#include "QMULset.h"
#include "HPset.h"
#include "EigenFaces.h"

#define QMUL_PATH "/Users/Mac-Robin/Documents/CompVis/Project/Project/QMUL"
#define HEAD_POSE_PATH "/Users/Mac-Robin/Documents/CompVis/Project/Project/HeadPoseImageDatabase"

using namespace cv;
using namespace std;

int main(int argc, const char * argv[]) {
    /*
    char* qmul = getenv("QMUL_PATH");
    char* hp = getenv("HEAD_POSE_PATH");

    if (!qmul) {
        cout << "QMUL_PATH environment variable is not defined" << endl;
        return 1;
    }
    if (!hp) {
        cout << "HEAD_POSE_PATH environment variable is not defined" << endl;
        return 1;
    }

    QMULset QMUL = QMULset(string(qmul));
    HPset HP = HPset(string(hp));
    */

    QMULset QMUL = QMULset(QMUL_PATH);
    HPset HP = HPset(HEAD_POSE_PATH);
    /*
    imshow("QMUL 0", QMUL.getAllImage(0));
    waitKey();

    imshow("HP 0", HP.getAllImage(1, 1));
    waitKey();
     */

    vector<Mat> set;
    QMUL.getPersonSet(1, set);
    //QMUL.getPoseSet(0, 90, set);
    answerQ4(set);

    return 0;
}
