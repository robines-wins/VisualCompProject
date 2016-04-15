
#include "answer.h"
#include <opencv2/opencv.hpp>
#include "EigenFaces.h"
#include "EigenFacePoseEstimation.h"

#define PATH_FOR_OUTPUT "/Users/Mac-Robin/Documents/CompVis/Project/Project/outputimages/"

using namespace cv;
using namespace std;

void answerQ3(vector<Mat> set){
    int numC[] = {1,2,5,10,20,50,100,200,500,1000,2000,3000,static_cast<int>(set.size())};
    for (int i = 0; i<13; i++) {
        cout<< numC[i] <<endl;
    }
    cout <<endl;
    cout << "k validation for reconstruction of training images" <<endl << endl;
    for (int i = 0; i< 13; i++) {
        cout << kFoldCrossValidationReconstruction(set, numC[i],7,false)<<endl;
    }
    cout <<endl;
    cout << "k validation for reconstruction of testing images" <<endl << endl;
    for (int i = 0; i< 13; i++) {
        cout << kFoldCrossValidationReconstruction(set, numC[i]) <<endl;
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
        normalize(base.row(i), toOutput,0,255,NORM_MINMAX);
        toOutput.reshape(1, 100).convertTo(toOutput, CV_8U);
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

void answerQ6(QMULset QMUL){

    int numC[] = {1,2,5,10,20,50,100,200,500,1000,2000,3000,3990};

    vector<Mat> S1,S2,S3, set;
    QMUL.getPersonSet(1, S1);
    QMUL.getPersonSet(2, S2);
    QMUL.getPersonSet(3, S3);

    set = S1;
    set.insert(set.end(), S2.begin(),S2.end());
    set.insert(set.end(), S3.begin(),S3.end());

    vector<double> labels;
    for (int i = 0; i<S1.size(); i++) {labels.push_back(1);}
    for (int i = 0; i<S2.size(); i++) {labels.push_back(2);}
    for (int i = 0; i<S3.size(); i++) {labels.push_back(3);}

    for (int i = 0; i<13; i++) {
        EigenRecognizerNorm ER(numC[i]);
        cout << kFoldCrossValidationRecognition(ER, set, labels, 7) << endl ;
    }


}

void answerQ8(QMULset QMUL, int optiNOC){
    vector<Mat> S1,S2,S3, set;
    QMUL.getPersonSet(1, S1);
    QMUL.getPersonSet(2, S2);
    QMUL.getPersonSet(3, S3);

    set = S1;
    set.insert(set.end(), S2.begin(),S2.end());
    set.insert(set.end(), S3.begin(),S3.end());

    vector<double> labels;
    for (int i = 0; i<S1.size(); i++) {labels.push_back(1);}
    for (int i = 0; i<S2.size(); i++) {labels.push_back(2);}
    for (int i = 0; i<S3.size(); i++) {labels.push_back(3);}

    EigenRecognizerProb ER(optiNOC);
    cout << kFoldCrossValidationRecognition(ER, set, labels, 7) << endl;

}

void answerQ16_1(QMULset qmul, HPset hp, vector<int> tiltClasses, vector<int> panClasses) {
    size_t numOfTiltClasses = tiltClasses.size();
    size_t numOfPanClasses = panClasses.size();
    size_t numberPoses = numOfTiltClasses * numOfPanClasses;
    Mat confusion = Mat::zeros(numberPoses, numberPoses, CV_32F);
    // Train the estimator for each pose use QMUL
    EigenFacePoseEstimator estimator(numberPoses, 30);
    for (size_t i = 0; i < numOfTiltClasses; i++) {
        for (size_t j = 0; j < numOfPanClasses; j++) {
            vector<Mat> coarsePoses;
            qmul.getCoarsePoseSet(tiltClasses, panClasses, i, j, coarsePoses);
            size_t index = i * numOfPanClasses + j;
            estimator.trainPose(index, coarsePoses);
            cout << "Trained pose " << index << " with tilt = " << tiltClasses[i] << " and pan = " << panClasses[j] << endl;
        }
    }
    // Estimate the poses from HP using the trained estimator
    for (size_t i = 0; i < numOfTiltClasses; i++) {
        for (size_t j = 0; j < numOfPanClasses; j++) {
            vector<Mat> coarsePoses;
            hp.getCoarsePoseSet(tiltClasses, panClasses, i, j, coarsePoses);
            size_t index = i * numOfPanClasses + j;
            for (size_t k = 0; k < coarsePoses.size(); k++) {
                size_t guess = estimator.estimatePose(coarsePoses[k]);
                confusion.at<float>(index, guess)++;
                cout << "Estimated pose " << index << " as " << guess << endl;
            }
            // Normalize the confusion row
            for (int col = 0; col < numberPoses; col++) {
                confusion.at<float>(index, col) /= coarsePoses.size();
            }
        }
    }
    // Pretty print the confusion matrix with fixed precision
    cout.setf(ios::fixed, ios::floatfield);
    cout.precision(2);
    cout << "[" << endl;
    for (size_t row = 0; row < numberPoses; row++) {
        cout << "    ";
        for (size_t col = 0; col < numberPoses; col++) {
            cout << confusion.at<float>(row, col) << "  ";
        }
        cout << endl;
    }
    cout << "]" << endl;
}
