#ifndef BOW_H
#define BOW_H

#include <vector>
#include <opencv2/opencv.hpp>
#include <opencv2/nonfree/nonfree.hpp>
#include "QMULset.h"

void TrainBoW(QMULset Dataset, Mat &codeBook, vector<vector<Mat>> &imageDescriptors, const int numCodewords);
void TestBoW(QMULset Dataset, const Mat codeBook, const vector<vector<Mat>> imageDescriptors);

#endif // BOW_H
