#include <opencv2/opencv.hpp>
#include "EigenFacePoseEstimation.h"

using namespace cv;
using namespace std;

EigenFacePoseEstimator::~EigenFacePoseEstimator() {
    for (size_t i = 0; i < poseRecognizers.size(); i++) {
        if (poseRecognizers[i] != NULL) {
            delete poseRecognizers[i];
        } 
    }
}

void EigenFacePoseEstimator::trainPose(int id, vector<Mat>& faces) {
    assert (id >= 0 && id < numberOfPoses);
    // Create and add the recognizer for the pose
    if (poseRecognizers[id] == NULL) {
        poseRecognizers[id] = new EigenRecognizerNorm(numberOfComponents);
    }
    // Train it, don't care about labels
    vector<double> labels;
    labels.resize(faces.size());
    poseRecognizers[id]->train(faces, labels);
}

int EigenFacePoseEstimator::estimatePose(Mat face) {
    double minDistance = numeric_limits<double>::max();
    int bestMatch = 0;
    for (int i = 0; i < numberOfPoses; i++) {
        double distance;
        double label = poseRecognizers[i]->labelise(face, distance);
        if (distance < minDistance) {
            minDistance = distance;
            bestMatch = i;
        }
    }
    return bestMatch;
}
