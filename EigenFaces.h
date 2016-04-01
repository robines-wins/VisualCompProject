//
//  EigenFaces.hpp
//  Project
//
//  Created by Robin Solignac on 31/03/2016.
//  Copyright © 2016 Robin Solignac. All rights reserved.
//

#ifndef EigenFaces_hpp
#define EigenFaces_hpp

#include <stdio.h>
using namespace cv;
using namespace std;
Mat imagesToPcaMatrix(vector<Mat> imgSet);

//warning row vector
Mat project(const Mat& vector, const Mat& base);

Mat backproject(const Mat& vector, const Mat& base);

double reconstructionError(Mat& traningM, Mat& testingM, int numOfComp);

double kFoldCrossValidation(vector<Mat> imgSet, int numOfComp, int k = 7);

#endif /* EigenFaces_hpp */
