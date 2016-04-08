#include <iostream>
#include <opencv2/opencv.hpp>
#include "EigenFaces.h"

using namespace cv;
using namespace std;

Mat imagesToPcaMatrix(vector<Mat> imgSet){
    Mat dataM = Mat(0, imgSet.front().rows*imgSet.front().cols, imgSet.front().type());
    for (auto it = imgSet.begin(); it != imgSet.end(); it++) {
        Mat vectorized = it->clone().reshape(0, 1);
        dataM.push_back(vectorized);
    }
    dataM.convertTo(dataM, CV_64F);
    return dataM;
}

//warning row vector
Mat project(const Mat& vector, const Mat& base){
    Mat projection = Mat(1, base.rows, base.type());
    assert(vector.rows == 1);
    
    for (int i =0; i<base.rows; i++) {
        //Mat temp = vector.mul(base.row(i));
        //projection.at<double>(0,i) = sum(temp)[0];
        projection.at<double>(0,i) = vector.dot(base.row(i));
        cout << projection<<endl;
    }
    return projection;
}

Mat backproject(const Mat& vector, const Mat& base){
    assert(vector.rows == 1);
    cout<<vector.type()<<" "<<base.type()<<endl;
    return vector*base;
}

double reconstructionError(Mat& traningM, Mat& testingM, int numOfComp){
    
    //caulation of mean vector, and substract it from each row vector
    Mat mean = Mat(1, traningM.cols, traningM.type());
    reduce(traningM, mean, 0, CV_REDUCE_AVG);
    
    for (int i=0; i<traningM.rows; i++) {
        traningM.row(i) -= mean;
    }
    
    Mat DtD, EValues, EVector;
    mulTransposed(traningM, DtD, true);
    eigen(DtD, EValues, EVector);
    
    Mat base;
    transpose(EVector.colRange(0, numOfComp), base);
    base.convertTo(base, traningM.type());
    cout << traningM.type() << endl << base.type() << endl;
    
    double reconstructionError = 0.0;
    
    for (int i = 0; i<testingM.rows; i++) {
        Mat temp,result;
        //project
        temp = project(testingM.row(i), base);
        //back
        result = backproject(temp, base);
        reconstructionError += norm(testingM.row(i),result,NORM_L2);
    }
    reconstructionError /= testingM.rows;
    return reconstructionError;
    
}

double kFoldCrossValidation(vector<Mat> imgSet, int numOfComp, int k){
    
    
    random_shuffle(imgSet.begin(), imgSet.end());
    double reconstructionerror = 0.0;
    
    Mat dataM = imagesToPcaMatrix(imgSet);
    assert(dataM.cols == imgSet.front().rows*imgSet.front().cols);
    assert(dataM.rows == imgSet.size());
    
    
    int imgPerFolds = (int)imgSet.size()/k;
    
    for (int i=0; i<k; i++) {
        Mat test,train;
        Mat trainA, trainB;
        
        dataM.rowRange(i*imgPerFolds, (i+1)*imgPerFolds).copyTo(test);
        //dataM(Range(i*imgPerFolds, (i+1)*imgPerFolds),Range(0, dataM.cols)).copyTo(test);
        assert(test.rows == imgPerFolds);
        
        if (i == 0) {
            dataM.rowRange((i+1)*imgPerFolds, k*imgPerFolds).copyTo(train);
            //dataM(Range((i+1)*imgPerFolds, k*imgPerFolds),Range(0, dataM.cols)).copyTo(train);
        }
        else if (i == k-1){
            dataM.rowRange(0, i*imgPerFolds).copyTo(train);
            //dataM(Range(0, i*imgPerFolds),Range(0, dataM.cols)).copyTo(train);
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
        
        cout << imgSet.front().channels() <<endl << dataM.channels()<<endl;
        reconstructionerror += reconstructionError(train, test, numOfComp);
    }
    
    reconstructionerror /= k;
    
    /*
     for (int i=0; i<dataM.rows; i++) {
     dataM.row(i) - mean;
     }*/
    
    
    
    
    
    
    return reconstructionerror;
    
    
}