#ifndef BOW_H
#define BOW_H

#include <vector>
#include <opencv2/opencv.hpp>
#include <opencv2/nonfree/nonfree.hpp>
#include "QMULset.h"

void kFoldsCrossValidation(QMULset Dataset, const int numCodewords);
void TrainBoW(vector< vector<Mat> > people_set,
              Mat &codeBook,
              vector<vector<Mat>> &imageDescriptors,
              const int numCodewords,
              const int partitionStart,
              const int partitionSize);
void TestBoW(vector< vector<Mat> > people_set,
             const Mat codeBook,
             const vector<vector<Mat>> imageDescriptors,
             const int partitionStart,
             const int partitionSize,
             double &recognition);

#endif // BOW_H
