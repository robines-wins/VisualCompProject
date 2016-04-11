#ifndef EigenFacePoseEstimation_h
#define EigenFacePoseEstimation_h

#include "EigenFaces.h"

using namespace cv;
using namespace std;

class EigenFacePoseEstimator {
private:
    int numberOfPoses;
    int numberOfComponents;
    vector<EigenRecognizerNorm*> poseRecognizers;
public:
    EigenFacePoseEstimator(int numberOfPoses, int numberOfComponents) :
            numberOfPoses(numberOfPoses), numberOfComponents(numberOfComponents) {
                poseRecognizers.resize(numberOfPoses, NULL);
            }
    ~EigenFacePoseEstimator();
    void trainPose(int id, vector<Mat>& faces);
    int estimatePose(Mat face);
};

#endif /* EigenFacePoseEstimation_h */
