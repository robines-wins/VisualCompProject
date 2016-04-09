#include <cstdlib>
#include <iostream>
#include <opencv2/opencv.hpp>

#include "main.h"
#include "QMULset.h"
#include "EigenFaces.h"

using namespace cv;
using namespace std;

int main(int argc, const char * argv[]) {
    char* qmul = getenv("QMUL_PATH");

    if (!qmul) {
        cout << "QMUL_PATH environment variable is not defined" << endl;
        return 1;
    }

    QMULset QMUL = QMULset(qmul);
    imshow("test",QMUL.get(0, -30, 10));
    waitKey();

    vector<Mat> set = QMUL.getSet(1);
    kFoldCrossValidation(set, 300);

    return 0;
}
