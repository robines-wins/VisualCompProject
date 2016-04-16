//
//  answer.h
//  Project
//
//  Created by Robin Solignac on 10/04/2016.
//  Copyright © 2016 Robin Solignac. All rights reserved.
//

#ifndef answer_h
#define answer_h

#include <stdio.h>
#include <opencv2/opencv.hpp>

#include "QMULset.h"
#include "HPset.h"

using namespace cv;
using namespace std;

void answerQ3(vector<Mat> set);

void answerQ4(vector<Mat> set);

void answerQ5(vector<Mat> set, int optimalfromQ3);

void answerQ6(QMULset QMUL);

void answerQ7(QMULset QMUL, int optiNOC);

void answerQ8(QMULset QMUL, int optiNOC);

void answerQ11(QMULset QMUL);

void answerQ16_1(QMULset qmul, HPset hp, vector<int> tiltClasses, vector<int> panClasses);

void answerQ16_2(QMULset qmul, HPset hp, vector<int> tiltClasses, vector<int> panClasses);

void answerQ16_3(QMULset qmul, HPset hp, vector<int> tiltClasses, vector<int> panClasses);

#endif /* answer_h */
