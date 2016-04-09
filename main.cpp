#include <cstdlib>
#include <iostream>
#include <opencv2/opencv.hpp>

#include "main.h"
#include "QMULset.h"
#include "HPset.h"
#include "EigenFaces.h"

using namespace cv;
using namespace std;

int main(int argc, const char * argv[]) {
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

    imshow("all", QMUL.allImageFromSubject(0));
    waitKey();

//    imshow("all", QMUL.allImageFromSubject(0));
//    waitKey();

//    vector<Mat> set = QMUL.getSet(1);
//    kFoldCrossValidation(set, 300);

    return 0;
}
