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

    imshow("QMUL YongminYGrey", QMUL.getAllImage("YongminYGrey"));
    waitKey();
    destroyWindow("QMUL YongminYGrey");

    imshow("HP Person15 series 2", HP.getAllImage(15, 2));
    waitKey();
    destroyWindow("HP Person15 series 2");

    vector<Mat> set;
    QMUL.getPersonSet(1, set);
    kFoldCrossValidation(set, 300);

    return 0;
}
