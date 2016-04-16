#include <cstdlib>
#include <iostream>
#include <opencv2/opencv.hpp>

#include "main.h"
#include "QMULset.h"
#include "HPset.h"
#include "answer.h"

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

    answerQ11(QMUL);

    HPset HP = HPset(string(hp));

    int coarseTilts[] = {-25, 0, 25};
    int coarsePans[] = {-90, -60, -30, 0, 30, 60, 90};
    vector<int> tiltClasses(begin(coarseTilts), end(coarseTilts));
    vector<int> panClasses(begin(coarsePans), end(coarsePans));

    answerQ16_1(QMUL, HP, tiltClasses, panClasses);

    return 0;
}
