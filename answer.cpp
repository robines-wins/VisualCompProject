
#include "answer.h"
#include <opencv2/opencv.hpp>
#include "EigenFaces.h"
#include "EigenFacePoseEstimation.h"

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

void answerQ16_1(QMULset qmul, HPset hp, vector<int> tiltClasses, vector<int> panClasses) {
    size_t numberPoses = tiltClasses.size() * panClasses.size();
    Mat confusion = Mat::zeros(numberPoses, numberPoses, CV_32SC1);
    // Train the estimator for each pose use QMUL
    EigenFacePoseEstimator estimator(numberPoses, 7);
    for (size_t i = 0; i < tiltClasses.size(); i++) {
        for (size_t j = 0; j < panClasses.size(); j++) {
            vector<Mat> coarsePoses;
            qmul.getCoarsePoseSet(tiltClasses, panClasses, i, j, coarsePoses);
            size_t index = i * panClasses.size() + j;
            estimator.trainPose(index, coarsePoses);
            cout << "Trained pose " << index << " with tilt = " << tiltClasses[i] << " and pan = " << panClasses[j] << endl;
        }
    }
    // Estimate the poses from HP using the trained estimator
    for (size_t i = 0; i < tiltClasses.size(); i++) {
        for (size_t j = 0; j < panClasses.size(); j++) {
            vector<Mat> coarsePoses;
            hp.getCoarsePoseSet(tiltClasses, panClasses, i, j, coarsePoses);
            size_t index = i * panClasses.size() + j;
            for (size_t k = 0; k < coarsePoses.size(); k++) {
                size_t guess = estimator.estimatePose(coarsePoses[k]);
                confusion.at<int>(index) = (int) guess;
                cout << "Estimated pose " << index << " as " << guess << endl;
            }
        }
    }
    // Print the confusion matrix
    cout << confusion << endl;
}
