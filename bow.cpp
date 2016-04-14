#include <string>
#include <algorithm>
#include <ctime>
#include <stdlib.h>
#include "bow.h"

#define DISPLAY 0
#define PEOPLE 31
#define PICTURES_PER_PERSON 133
#define TESTING 5
#define K 7

/* k = 7 */
void kFoldsCrossValidation(QMULset Dataset, const int numCodewords) {
    // randomly shuffle the input data set
    srand(time(0));
    vector< vector<Mat> > people_set((PEOPLE));
    for (unsigned int it = 0; it < PEOPLE; it++) {
        Dataset.getPersonSet(it, people_set[it]);
        random_shuffle(people_set[it].begin(), people_set[it].end());
    }

    // validate for k = 7 fold
    int partition = PICTURES_PER_PERSON / K;
    double avg_recognition = 0;
    for (unsigned int it = 0; it < K; it++) {
        Mat codeBook;
        vector<vector<Mat>> imageDescriptors;
        int partitionStart = it * partition;
        TrainBoW(people_set, codeBook, imageDescriptors, numCodewords, partitionStart, partition);
        double recognition;
        TestBoW(people_set, codeBook, imageDescriptors, partitionStart, partition, recognition);
        avg_recognition += recognition;
    }
    avg_recognition /= K;

    cout << numCodewords << ": " << avg_recognition * 100 << "%" << endl;
}

/* Train BoW */
void TrainBoW(vector< vector<Mat> > people_set,
              Mat &codeBook,
              vector<vector<Mat>> &imageDescriptors,
              const int numCodewords,
              const int partitionStart,
              const int partitionSize)
{
    // Create a SIFT feature detector object
    Ptr<FeatureDetector> featureDetector = FeatureDetector::create("SIFT");

    // Create a SIFT descriptor extractor object
    Ptr<DescriptorExtractor> descriptorExtractor = DescriptorExtractor::create("SIFT");

    // Mat object to store all the SIFT descriptors of all training categories
    Mat D;

    // loop for each person
    for (unsigned int i = 0; i < people_set.size(); i++) {
        // each image of each category in a partition
        for (unsigned int j = (partitionStart == 0) ? partitionSize : 0; j < people_set[i].size(); j++) {
            if (j == partitionStart) {
                j += partitionSize-1;
                continue;
            }

            Mat I = people_set[i][j];

            // detect key points
            vector<KeyPoint> keyPoints;
            featureDetector->detect(I, keyPoints);

#if DISPLAY
            if (j == 15) {
                // test to see the key points on the image
                Mat outI;
                drawKeypoints(I, keyPoints, outI);
                CvScalar color = {0.0, 0.0, 255.0, 0.0};
                rectangle(outI, annotation, color);
                imshow("Annotated test with key points", outI);
                waitKey(0);
            }
#endif

            // compute SIFT descriptors
            Mat descriptor;
            descriptorExtractor->compute(I, keyPoints, descriptor);

            // Add descriptors to D
            D.push_back(descriptor);
        }
    }

    cout << "computed descriptors, creating the codebook" << endl;

    // create a codebook
    BOWKMeansTrainer trainer = BOWKMeansTrainer(numCodewords);
    trainer.add(D);
    codeBook = trainer.cluster();

    cout << "clustered, generating histograms" << endl;

    // Represent the object categories using the codebook
    // BOW histogram for each image
    Ptr<DescriptorMatcher> descriptorMatcher = DescriptorMatcher::create("BruteForce");
    Ptr<BOWImgDescriptorExtractor> bowDExtractor = new BOWImgDescriptorExtractor(descriptorExtractor, descriptorMatcher);
    bowDExtractor->setVocabulary(codeBook);

    // loop for each category
    for (unsigned int i = 0; i < people_set.size(); i++) {
        // each image of each category
        vector<Mat> category_descriptors;

        for (unsigned int j = (partitionStart == 0) ? partitionSize : 0; j < people_set[i].size()-TESTING; j++) {
            if (j == partitionStart) {
                j += partitionSize-1;
                continue;
            }

            Mat I = people_set[i][j];

            // detect key points
            vector<KeyPoint> keyPoints;
            featureDetector->detect(I, keyPoints);

            // Compute histogram representation and store in descriptors
            Mat histogram;
            bowDExtractor->compute2(I, keyPoints, histogram);
            category_descriptors.push_back(histogram);
        }
        imageDescriptors.push_back(category_descriptors);
    }
}

/* Test BoW */
void TestBoW(vector< vector<Mat> > people_set,
             const Mat codeBook,
             const vector<vector<Mat>> imageDescriptors,
             const int partitionStart,
             const int partitionSize,
             double &recognition)
{
    // Create a SIFT feature detector object
    Ptr<FeatureDetector> featureDetector = FeatureDetector::create("SIFT");

    // Create a SIFT descriptor extractor object
    Ptr<DescriptorExtractor> descriptorExtractor = DescriptorExtractor::create("SIFT");

    // BOW histogram for each image
    Ptr<DescriptorMatcher> descriptorMatcher = DescriptorMatcher::create("BruteForce");
    Ptr<BOWImgDescriptorExtractor> bowDExtractor = new BOWImgDescriptorExtractor(descriptorExtractor, descriptorMatcher);

    bowDExtractor->setVocabulary(codeBook);

    int numCorrect = 0;

    // loop for each category
    for (unsigned int i = 0; i < people_set.size(); i++) {
        // each image of each category
        for (unsigned int j = partitionStart; j < partitionStart+partitionSize; j++) {
            Mat I = people_set[i][j];

            // detect key points
            vector<KeyPoint> keyPoints;
            featureDetector->detect(I, keyPoints);

            // Compute histogram representation
            Mat histogram;
            bowDExtractor->compute2(I, keyPoints, histogram);

            // compare and find the best matching histogram
            double best_dist = numeric_limits<double>::max();
            unsigned int best_m = 0;
            for (unsigned int m = 0; m < imageDescriptors.size(); m++) {
                for (unsigned int n = 0; n < imageDescriptors[m].size(); n++) {
                    // use chi square distance to compare histograms
                    Mat diff = histogram - imageDescriptors[m][n];
                    Mat numerator = diff.mul(diff);
                    Mat denominator = 1 / (histogram + imageDescriptors[m][n]);
                    double dist = sum(numerator.mul(denominator))[0];

                    if (dist < best_dist) {
                        best_dist = dist;
                        best_m = m;
                    }
                }
            }

            // assign the category index
            if (best_m == i) numCorrect++;
        }
    }

    recognition = double(numCorrect) / double(PEOPLE * TESTING);
//    cout << ratio * 100 << "% recognition" << endl;
}

