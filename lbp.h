#ifndef LBP_H
#define LBP_H

#include <opencv2/opencv.hpp>
#include <math.h>

using namespace cv;
using namespace std;

const double pi = 3.1415926;

Mat createSpatialPyramidHistogram(Mat image, int Level);

Mat createLbpHistorgram(Mat image, int numGridHorizontal, int numGridVertical);

double calculateChiSquare(Mat histogram1, Mat histogram2);

double spatialLevelOf(Mat histogram);

double calculateSpatialChi(Mat histogram1, Mat histogram2);

void TrainLbp(vector< vector<Mat> >& people_set,
        vector<vector <Mat> > &imageDescriptors,
        const int levels,
        const int partitionStart,
        const int partitionSize);

void TestLbp(vector< vector<Mat> >& people_set,
             const vector<vector <Mat> >& imageDescriptors,
             const int levels,
             const int partitionStart,
             const int partitionSize,
             double &recognition);

void TrainLbpProb(vector< vector<Mat> >& people_set,
                  vector<vector<Mat> > &imageDescriptors,
                  const int levels,
                  const int partitionStart,
                  const int partitionSize,
                  vector<Mat> &covar_mat,
                  vector<Mat> &mean_vectors);

void TestLbpProb(vector< vector<Mat> >& people_set,
                 const int levels,
                 const int partitionStart,
                 const int partitionSize,
                 const vector<Mat>& covar,
                 const vector<Mat>& mean_vector,
                 double &recognition);

double LbpkFoldsCrossValidation(vector< vector<Mat> >& people_set, const int levels, int k, bool prob);

#endif
