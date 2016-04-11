#include <iostream>
#include <opencv2/opencv.hpp>
#include "EigenFaces.h"

using namespace cv;
using namespace std;
//WARNING: all of this function are creating/working with row vector



Mat imagesToPcaMatrix(vector<Mat> imgSet){
    Mat dataM = Mat(0, imgSet.front().rows*imgSet.front().cols, imgSet.front().type());
    for (auto it = imgSet.begin(); it != imgSet.end(); it++) {
        Mat vectorized = it->clone().reshape(0, 1);
        dataM.push_back(vectorized);
    }
    dataM.convertTo(dataM, CV_64F);
    return dataM;
}

Mat project(const Mat& vectors, const Mat& base){
    Mat projections = Mat(vectors.rows, base.rows, base.type());
    
    for (int r = 0; r<vectors.rows; r++) {
        for (int i =0; i<base.rows; i++) { //compute each coordinate according to the course formula
            projections.at<double>(r,i) = vectors.row(r).dot(base.row(i));
        }
    }
    
    return projections;
}

Mat backproject(const Mat& vectors, const Mat& base){
    return vectors*base; //just a base transition because here we have the transition matrix
}

Mat computeEigenBase(Mat& data, int numOfComp){
    Mat mean, EValues, EVector, covar;
    calcCovarMatrix(data, covar, mean, CV_COVAR_NORMAL | CV_COVAR_ROWS);
    eigen(covar, EValues, EVector);
    
    //if (inri) {outputEVectors(EVector);}
    
    Mat base;
    transpose(EVector.colRange(0, numOfComp), base); //transpose matrix eigenvector to get back to row vector and selecting only the number we want
    base.convertTo(base, data.type()); //make sure type of base and vector are the same
    
    return base;
}

double reconstructionError(Mat& traningM, Mat& testingM, int numOfComp){
    
    //calculation of mean vector, and substract it from each row vector
    Mat mean = Mat(1, traningM.cols, traningM.type());
    reduce(traningM, mean, 0, CV_REDUCE_AVG);
    
    for (int i=0; i<traningM.rows; i++) {
        traningM.row(i) -= mean;
        
    }
    
    
    Mat base = computeEigenBase(traningM, numOfComp);
    
    double reconstructionError = 0.0;
    
    for (int i = 0; i<testingM.rows; i++) {
        Mat temp,result;
        //project
        temp = project(testingM.row(i), base);
        //back
        result = backproject(temp, base);
        
        reconstructionError += norm(testingM.row(i),result,NORM_L2);
    }
    reconstructionError = reconstructionError/testingM.rows;
    return reconstructionError;
    
}



double kFoldCrossValidationReconstruction(vector<Mat> imgSet, int numOfComp, int k,bool testWithtest){
    
    
    random_shuffle(imgSet.begin(), imgSet.end()); //shuffle the order of images
    double reconstructionerror = 0.0; //initiate the error
    
    Mat dataM = imagesToPcaMatrix(imgSet); //transform vector of image to matrix of row vector
    assert(dataM.cols == imgSet.front().rows*imgSet.front().cols);
    assert(dataM.rows == imgSet.size());
    
    for (int i=0; i<k; i++) { //for each fold
        Mat test,train;
        splitM(dataM, test, train, k, i);
        //compute the reconstruction error for this configuration
        reconstructionerror += testWithtest ? reconstructionError(train, test, numOfComp) : reconstructionError(train, train, numOfComp);
    }
    
    //transform sum of error to an average
    reconstructionerror = reconstructionerror/(double)k;
    
    return reconstructionerror;
    
    
}

double recognitionRate(EigenRecognizer& ER, vector<Mat>& testImages, vector<double>& labels){
    assert(testImages.size() == labels.size());
    int rate = 0.0;
    
    for (int i=0; i<testImages.size(); i++) {
        if (labels[i] == ER.labelise(testImages[i])) {
            rate++;
        }
    }
    return (double)rate/(double)testImages.size();
}

double kFoldCrossValidationRecognition(EigenRecognizer& ER,vector<Mat> imgSet, vector<double> labels, int k){
    vector<int> indexs = randomIndexes(imgSet.size());
    double recognition = 0.0;
    
    vector<Mat> train,test;
    vector<double> trainl,testl;
    uint imgperfold = imgSet.size()/k;
    
    for (int i = 0; i<k; i++) {
        
        for (int j=0; j<imgSet.size(); j++) {
            if (j>=i*imgperfold && j<(i+1)*imgperfold) {
                test.push_back(imgSet[indexs[j]]);
                testl.push_back(labels[indexs[j]]);
            }
            else{
                train.push_back(imgSet[indexs[j]]);
                trainl.push_back(labels[indexs[j]]);
            }
        }
        
        ER.train(train, trainl);
        recognition += recognitionRate(ER, test, testl);
        
    }
    
    return recognition/(double)k;
}

void splitM(Mat& dataM, Mat& test, Mat& train,int k, int i){
    int imgPerFolds = (int)dataM.rows/k;
    Mat trainA, trainB;
    
    dataM.rowRange(i*imgPerFolds, (i+1)*imgPerFolds).copyTo(test); //create the matrix of test vector
    assert(test.rows == imgPerFolds);
    
    //create matrxix of training vector specila case on extraction of vectores with 1st and last fold
    if (i == 0) {
        dataM.rowRange((i+1)*imgPerFolds, k*imgPerFolds).copyTo(train);
    }
    else if (i == k-1){
        dataM.rowRange(0, i*imgPerFolds).copyTo(train);
    }
    else{
        dataM.rowRange(0, i*imgPerFolds).copyTo(trainA);
        dataM.rowRange((i+1)*imgPerFolds, k*imgPerFolds).copyTo(trainB);
        dataM(Range(0, i*imgPerFolds),Range(0, dataM.cols)).copyTo(trainA);
        dataM(Range((i+1)*imgPerFolds, k*imgPerFolds),Range(0, dataM.cols)).copyTo(trainB);
        vconcat(trainA, trainB, train);
    }
    
    assert(train.cols == dataM.cols);
    assert(train.rows == (k-1)*imgPerFolds);
    assert(test.rows == imgPerFolds);
    
}


vector<int> randomIndexes(int size){
    vector<int> randi = vector<int>(size);
    for (int i=0; i<size; i++) {
        randi[i] = i;
    }
    randShuffle(randi);
    return randi;
}


void EigenRecognizerNorm::train(vector<Mat>& images, vector<double>& labels){
    Mat data = imagesToPcaMatrix(images);
    base = computeEigenBase(data, noc);
    trained = project(data, base);
    labels = labels;
}

double EigenRecognizerNorm::labelise(Mat image){
    Mat projection = project(image.reshape(1, 1),base);
    int nearestIndex = 0;
    double smallestnorm = numeric_limits<double>::max();
    
    for (int i = 0; i<trained.rows; i++) {
        double tnorm = norm(projection, trained.row(i));
        if (tnorm<smallestnorm) {
            smallestnorm = tnorm;
            nearestIndex = i;
        }
    }
    
    return labels[nearestIndex];
}

void EigenRecognizerProb::train(vector<Mat>& images, vector<double>& labels){
    Mat dataM = imagesToPcaMatrix(images);
    Mat labelM = Mat(labels.size(), 1, CV_64F);
    for (int i=0; i<labels.size(); i++) {
        labelM.at<double>(i, 0) = labels[i];
    }
    
    NGC.train(dataM, labelM);
}

double EigenRecognizerProb::labelise(Mat& image){
    return NGC.predict(image.reshape(image.channels(), 1));
    
}













