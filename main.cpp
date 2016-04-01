#include <iostream>
#include <opencv2/opencv.hpp>

#include "main.h"
#include "QMULset.h"
#include "EigenFaces.h"

#define QMUL_PATH_ROBIN "/Users/Mac-Robin/Documents/CompVis/Project/Project/QMUL"

using namespace cv;
using namespace std;





int main(int argc, const char * argv[]) {
    
    QMULset QMUL = QMULset(QMUL_PATH_ROBIN);
    //imshow("test",QMUL.get(0, -30, 10));
    //waitKey();
    
    vector<Mat> set = QMUL.getSet(1);
    kFoldCrossValidation(set, 300);
    
    
    
    return 0;
}

