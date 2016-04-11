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

    imshow("QMUL 0", QMUL.getAllImage(0));
    waitKey();

    imshow("HP 0", HP.getAllImage(1, 1));
    waitKey();

    vector<Mat> set;
    QMUL.getPersonSet(1, set);
    //QMUL.getPoseSet(0, 90, set);
    answerQ4(set);

    return 0;
}
