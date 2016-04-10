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

Mat computeEigenBase(Mat& data, int numOfComp);

double reconstructionError(Mat& traningM, Mat& testingM, int numOfComp);

double kFoldCrossValidation(vector<Mat> imgSet, int numOfComp, int k = 7, bool testWithtest = true);

void splitM(Mat& dataM, Mat& test, Mat& train,int k, int i);

void anwswerQ3(vector<Mat> set);

void anwswerQ4(vector<Mat> set);

void answerQ5(vector<Mat> set, int optimalfromQ3);

#endif /* EigenFaces_hpp */
