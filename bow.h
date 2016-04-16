#ifndef BOW_H
#define BOW_H

#include <vector>
#include <opencv2/opencv.hpp>
#include <opencv2/nonfree/nonfree.hpp>
#include "QMULset.h"

void BoWkFoldsCrossValidation(QMULset Dataset, const int numCodewords, bool prob);
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
void TrainBoWProb(vector< vector<Mat> > people_set,
                  Mat &codeBook,
                  vector<vector<Mat>> &imageDescriptors,
                  const int numCodewords,
                  const int partitionStart,
                  const int partitionSize,
                  vector<Mat> &covar_mat,
                  vector<Mat> &mean_vectors);
void TestBoWProb(vector< vector<Mat> > people_set,
                 const Mat codeBook,
                 const int partitionStart,
                 const int partitionSize,
                 const vector<Mat> covar,
                 const vector<Mat> mean_vector,
                 double &recognition);

double FindBestBoWDistance(Mat pose,
          const Mat codeBook,
          const vector<vector<Mat>> imageDescriptors);

#endif // BOW_H
