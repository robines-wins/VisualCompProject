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

double kFoldCrossValidationReconstruction(vector<Mat> imgSet, int numOfComp, int k = 7, bool testWithtest = true);



void splitM(Mat& dataM, Mat& test, Mat& train,int k, int i);

vector<int> randomIndexes(int size);

class EigenRecognizer{
public:
    virtual ~EigenRecognizer() {}
    virtual void train(vector<Mat>& images, vector<double>& labels) =0;
    virtual double labelise(Mat& image) =0;
};

class EigenRecognizerNorm : public EigenRecognizer{
private:
    Mat base;
    Mat trained;
    cv::vector<double> labels;
    int noc;
public:
    EigenRecognizerNorm(int numOfComp){noc = numOfComp;}
    void train(vector<Mat>& images, vector<double>& labels);
    double labelise(Mat& image);
    double labelise(Mat& image, double& distance);
};

class normalGaussian{
private:
    Mat mean;
    Mat covarDiag;
    double det;
public:
    normalGaussian(Mat& data);
    double probOf(Mat& vector);
};

class EigenRecognizerProb : public EigenRecognizer{
private:
    int noc;
    map<double,normalGaussian> NGC;
public:
    EigenRecognizerProb(int numOfComp){noc = numOfComp;}
    void train(std::vector<Mat>& images, std::vector<double>& labels);
    double labelise(Mat& image);
};



double recognitionRate(EigenRecognizer& ER, vector<Mat>& testImages, vector<double>& labels);

double kFoldCrossValidationRecognition(EigenRecognizer& ER,vector<Mat> imgSet, vector<double> labels, int k);

#endif /* EigenFaces_hpp */
