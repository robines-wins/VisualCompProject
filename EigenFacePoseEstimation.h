#ifndef EigenFacePoseEstimation_h
#define EigenFacePoseEstimation_h

#include "EigenFaces.h"

using namespace cv;
using namespace std;

class EigenFacePoseEstimator {
private:
    vector<EigenRecognizer> poseRecognizers;
public:
    int trainPose(vector<Mat>& faces);
    int estimatePose(Mat face);
};

#endif /* EigenFacePoseEstimation_h */
