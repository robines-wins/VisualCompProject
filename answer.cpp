#include "answer.h"
#include <opencv2/opencv.hpp>
#include "EigenFaces.h"
#include "lbp.h"
#include "bow.h"
#include "EigenFacePoseEstimation.h"

#define PATH_FOR_OUTPUT "/Users/Mac-Robin/Documents/CompVis/Project/Project/outputimages/"

using namespace cv;
using namespace std;

void answerQ3(vector<Mat> set){
    int numC[] = {1,2,5,10,20,50,100,200,500,1000,2000,3000,static_cast<int>(set.size())};
    for (int i = 0; i<13 && (numC[i]<=static_cast<int>(set.size()*6)/7); i++) {
        cout<< numC[i] <<endl;
    }
    cout <<endl;
    cout << "k validation for reconstruction of training images" <<endl << endl;
    for (int i = 0; i<13 && (numC[i]<=static_cast<int>(set.size()*6)/7); i++) {
        cout << kFoldCrossValidationReconstruction(set, numC[i],7,false)<<endl;
    }
    cout <<endl;
    cout << "k validation for reconstruction of testing images" <<endl << endl;
    for (int i = 0; i<13 && (numC[i]<=static_cast<int>(set.size()*6)/7); i++) {
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
    imwrite(path + "means.png", mean);

    Mat base = computeEigenBase(train, 10);
    for (int i = 0; i<10; i++) {
        Mat toOutput;
        normalize(base.row(i), toOutput,0,255,NORM_MINMAX);
        toOutput.reshape(1, 100).convertTo(toOutput, CV_8U);
        imwrite(path + "Evector" +to_string(i)+".png", toOutput);
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
            imwrite(path + "random" +to_string(j)+ "_Original.png", toOutput);

            toOutput = backproject(project(randImg[j], base), base);
            normalize(toOutput, toOutput,0,255,NORM_MINMAX);
            toOutput = toOutput.reshape(1, 100);
            toOutput.convertTo(toOutput, CV_8U);
            imwrite(path + "random" +to_string(j)+ "_reconstruct_" +to_string(numOfComp[i])+"vectors.png", toOutput);
        }
    }
}

void answerQ6(QMULset QMUL){

    int numC[] = {1,2,5,10,20,50,100,200,500,1000,2000,3000,4123};

    vector<Mat> S1,S2,S3,S4, set;
    QMUL.getPersonSet(1, S1);
    QMUL.getPersonSet(2, S2);
    QMUL.getPersonSet(3, S3);
    QMUL.getPersonSet(4, S4);

    set = S1;
    set.insert(set.end(), S2.begin(),S2.end());
    set.insert(set.end(), S3.begin(),S3.end());
    set.insert(set.end(), S4.begin(),S4.end());

    vector<double> labels;
    for (int i = 0; i<S1.size(); i++) {labels.push_back(1);}
    for (int i = 0; i<S2.size(); i++) {labels.push_back(2);}
    for (int i = 0; i<S3.size(); i++) {labels.push_back(3);}
    for (int i = 0; i<S4.size(); i++) {labels.push_back(4);}

    for (int i = 0; i<13; i++) {
        EigenRecognizerNorm ER(numC[i]);
        cout << kFoldCrossValidationRecognition(ER, set, labels, 7) << endl ;
    }


}

void answerQ7(QMULset QMUL, int optiNOC){
    EigenRecognizerNorm ER(optiNOC);

    vector<Mat> S1,S2,S3,S4, set;
    QMUL.getPersonSet(1, S1);
    QMUL.getPersonSet(2, S2);
    QMUL.getPersonSet(3, S3);
    QMUL.getPersonSet(4, S4);

    set = S1;
    set.insert(set.end(), S2.begin(),S2.end());
    set.insert(set.end(), S3.begin(),S3.end());
    set.insert(set.end(), S4.begin(),S4.end());

    vector<double> labels;
    for (int i = 0; i<S1.size(); i++) {labels.push_back(1);}
    for (int i = 0; i<S2.size(); i++) {labels.push_back(2);}
    for (int i = 0; i<S3.size(); i++) {labels.push_back(3);}
    for (int i = 0; i<S4.size(); i++) {labels.push_back(4);}


    vector<int> indexs = randomIndexes(set.size());

    int i = rand()/7;

    uint imgperfold = set.size()/7;

    vector<Mat> train,test;
    vector<double> trainl,testl;

    for (int j=0; j<set.size(); j++) {
        if (j>=i*imgperfold && j<(i+1)*imgperfold) {
            test.push_back(set[indexs[j]]);
            testl.push_back(labels[indexs[j]]);
        }
        else{
            train.push_back(set[indexs[j]]);
            trainl.push_back(labels[indexs[j]]);
        }
    }

    ER.train(train, trainl);

    int good = 0, bad =0;
    string path = PATH_FOR_OUTPUT;
    for (int j=0; j<test.size() && good<2 && bad<2; j++) {
        if (testl[j] == ER.labelise(test[j])) {
            good++;
            imwrite(path + "goodLabelise_" + to_string(good) +".bmp" , testl[j]);
        }
    }

}

void answerQ8(QMULset QMUL, int optiNOC){
    vector<Mat> S1,S2,S3,S4, set;
    QMUL.getPersonSet(1, S1);
    QMUL.getPersonSet(2, S2);
    QMUL.getPersonSet(3, S3);
    QMUL.getPersonSet(4, S4);

    set = S1;
    set.insert(set.end(), S2.begin(),S2.end());
    set.insert(set.end(), S3.begin(),S3.end());
    set.insert(set.end(), S4.begin(),S4.end());

    vector<double> labels;
    for (int i = 0; i<S1.size(); i++) {labels.push_back(1);}
    for (int i = 0; i<S2.size(); i++) {labels.push_back(2);}
    for (int i = 0; i<S3.size(); i++) {labels.push_back(3);}
    for (int i = 0; i<S4.size(); i++) {labels.push_back(4);}

    EigenRecognizerProb ER(optiNOC);
    cout << kFoldCrossValidationRecognition(ER, set, labels, 7) << endl;

}

void prettyPrintMatrix(Mat m) {
    // Pretty print the confusion matrix with fixed precision
    cout.setf(ios::fixed, ios::floatfield);
    cout.precision(2);
    cout << "[" << endl;
    for (size_t row = 0; row < m.rows; row++) {
        cout << "    ";
        for (size_t col = 0; col < m.cols; col++) {
            cout << m.at<float>(row, col) << "  ";
        }
        cout << endl;
    }
    cout << "]" << endl;
}

void answerQ11(QMULset QMUL) {
    vector<vector<Mat>> allPeople;
    allPeople.resize(QMUL.peopleCount());
    for (int i = 0; i < QMUL.peopleCount(); i++) {
        QMUL.getPersonSet(i, allPeople[i]);
    }
    cout << "Regular: " << LbpkFoldsCrossValidation(allPeople, 2, 7, true) << endl;
    cout << "Probabilistic: " << LbpkFoldsCrossValidation(allPeople, 2, 7, true) << endl;
}

void answerQ16_1(QMULset qmul, HPset hp, vector<int> tiltClasses, vector<int> panClasses) {
    size_t numOfTiltClasses = tiltClasses.size();
    size_t numOfPanClasses = panClasses.size();
    size_t numberPoses = numOfTiltClasses * numOfPanClasses;
    Mat confusion = Mat::zeros(numberPoses, numberPoses, CV_32F);
    // Train the estimator for each pose use QMUL
    EigenFacePoseEstimator estimator(numberPoses, 120);
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
    prettyPrintMatrix(confusion);
}

void answerQ16_2(QMULset qmul, HPset hp, vector<int> tiltClasses, vector<int> panClasses) {
    size_t numOfTiltClasses = tiltClasses.size();
    size_t numOfPanClasses = panClasses.size();
    size_t numberPoses = numOfTiltClasses * numOfPanClasses;
    Mat confusion = Mat::zeros(numberPoses, numberPoses, CV_32F);
    // Create a set of poses
    vector<vector<Mat>> poses;
    for (size_t i = 0; i < numOfTiltClasses; i++) {
        for (size_t j = 0; j < numOfPanClasses; j++) {
            vector<Mat> coarsePoses;
            qmul.getCoarsePoseSet(tiltClasses, panClasses, i, j, coarsePoses);
            poses.push_back(coarsePoses);
        }
    }
    // Train LBP on poses instead of persons
    int levels = 2;
    vector<vector<Mat>> imageDescriptors;
    TrainLbp(poses, imageDescriptors, levels, -1, 0);
    // Estimate the poses from HP using the trained poses
    for (size_t i = 0; i < numOfTiltClasses; i++) {
        for (size_t j = 0; j < numOfPanClasses; j++) {
            vector<Mat> coarsePoses;
            hp.getCoarsePoseSet(tiltClasses, panClasses, i, j, coarsePoses);
            size_t index = i * numOfPanClasses + j;
            for (size_t k = 0; k < coarsePoses.size(); k++) {
                size_t guess = FindBestLbpMatch(coarsePoses[k], imageDescriptors, levels);
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
    prettyPrintMatrix(confusion);
}

void answerQ16_3(QMULset qmul, HPset hp, vector<int> tiltClasses, vector<int> panClasses) {
    size_t numOfTiltClasses = tiltClasses.size();
    size_t numOfPanClasses = panClasses.size();
    size_t numberPoses = numOfTiltClasses * numOfPanClasses;
    Mat confusion = Mat::zeros(numberPoses, numberPoses, CV_32F);
    // Train one BoW per pose
    int codewords = 900;
    vector<Mat> codeBooks;
    codeBooks.resize(numberPoses);
    vector<vector<vector<Mat>>> imageDescriptors;
    imageDescriptors.resize(numberPoses);
    for (size_t i = 0; i < numOfTiltClasses; i++) {
        for (size_t j = 0; j < numOfPanClasses; j++) {
            vector<Mat> coarsePoses;
            qmul.getCoarsePoseSet(tiltClasses, panClasses, i, j, coarsePoses);
            vector<vector<Mat>> poses;
            poses.push_back(coarsePoses);
            size_t index = i * numOfPanClasses + j;
            TrainBoW(poses, codeBooks[index], imageDescriptors[index], codewords, -1, 0);
        }
    }
    // Estimate the poses from HP using the trained poses
    for (size_t i = 0; i < numOfTiltClasses; i++) {
        for (size_t j = 0; j < numOfPanClasses; j++) {
            vector<Mat> coarsePoses;
            hp.getCoarsePoseSet(tiltClasses, panClasses, i, j, coarsePoses);
            size_t index = i * numOfPanClasses + j;
            for (size_t k = 0; k < coarsePoses.size(); k++) {
                // Try each BoW on the pose, find the closest one as the guess
                double best_dist = numeric_limits<double>::max();
                size_t best_bow = -1;
                for (size_t bow = 0; bow < numberPoses; bow++) {
                    double dist = FindBestBoWDistance(coarsePoses[k], codeBooks[bow], imageDescriptors[bow]);
                    if (dist < best_dist) {
                        best_dist = dist;
                        best_bow = bow;
                    }
                }
                confusion.at<float>(index, best_bow)++;
                cout << "Estimated pose " << index << " as " << best_bow << endl;
            }
            // Normalize the confusion row
            for (int col = 0; col < numberPoses; col++) {
                confusion.at<float>(index, col) /= coarsePoses.size();
            }
        }
    }
    // Pretty print the confusion matrix with fixed precision
    prettyPrintMatrix(confusion);
}
