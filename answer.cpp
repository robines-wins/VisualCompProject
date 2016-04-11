
#include "answer.hpp"
#include <opencv2/opencv.hpp>
#include "EigenFaces.h"

#define PATH_FOR_OUTPUT "/Users/Mac-Robin/Documents/CompVis/Project/Project/outputimages/"

using namespace cv;
using namespace std;

void answerQ3(vector<Mat> set){
    
    int numC[] = {1,2,5,10,20,50,100,200,500,1000,2000,3000,set.front().rows*set.front().cols};
    for (int i = 0; i<13; i++) {
        cout<< numC[i] <<endl;
    }
    cout <<endl;
    for (int i = 0; i< 13; i++) {
        cout << "k validation for reconstruction of training images" <<endl << endl;
        cout << kFoldCrossValidationReconstruction(set, numC[i],7,false)<<endl;
    }
    cout <<endl;
    for (int i = 0; i< 13; i++) {
        cout << "k validation for reconstruction of testing images" <<endl << endl;
        cout << numC[i] <<" " << kFoldCrossValidationReconstruction(set, numC[i]) <<endl;
    }
}

void answerQ4(vector<Mat> set){
    random_shuffle(set.begin(), set.end());
    Mat dataM = imagesToPcaMatrix(set);
    Mat test,train;
    splitM(dataM, test, train, 7, rand()%7);
    
    Mat mean = Mat(1, train.cols, train.type());
    reduce(train, mean, 0, CV_REDUCE_AVG);
    mean.reshape(1, 100).convertTo(mean, CV_8U);
    string path = PATH_FOR_OUTPUT;
    imwrite(path + "means.bmp", mean);
    
    Mat base = computeEigenBase(train, 10);
    for (int i = 0; i<10; i++) {
        Mat toOutput;
        base.row(i).reshape(1, 100).convertTo(toOutput, CV_8U);
        cout << "plop" + to_string(i)<<endl;
        imwrite(path + "Evector" +to_string(i)+".bmp", toOutput);
    }
    
}

void answerQ5(vector<Mat> set, int optimalfromQ3){
    Mat dataM = imagesToPcaMatrix(set);
    Mat test,train;
    splitM(dataM, test, train, 7, rand()%7);
    
    int numOfComp[] = {1,5,optimalfromQ3};
    Mat randImg[4] = {train.row(rand()%train.rows), train.row(rand()%train.rows), test.row(rand()%test.rows), test.row(rand()%test.rows)};
    
    for (int i=0; i<3; i++) {
        Mat base = computeEigenBase(train, numOfComp[i]);
        for (int j=0; j<4; j++) {
            Mat toOutput = randImg[j].reshape(1, 100);
            toOutput.convertTo(toOutput, CV_8U);
            string path = PATH_FOR_OUTPUT;
            imwrite(path + "random" +to_string(j)+ "_Original_" +to_string(numOfComp[i])+"vectors.bmp", toOutput);
            
            toOutput = backproject(project(randImg[j], base), base).reshape(1, 100);
            toOutput.convertTo(toOutput, CV_8U);
            imwrite(path + "random" +to_string(j)+ "_reconstruct_" +to_string(numOfComp[i])+"vectors.bmp", toOutput);
        }
    }
    
    
}