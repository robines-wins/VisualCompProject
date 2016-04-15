#include <string>
#include <algorithm>
#include <ctime>
#include <stdlib.h>
#include <iostream>
#include <fstream>
#include "bow.h"

#define DISPLAY 0
#define PEOPLE 31
#define PICTURES_PER_PERSON 133
#define K 7

/* k = 7 */
void BoWkFoldsCrossValidation(QMULset Dataset, const int numCodewords, bool prob) {
    // initialize non free module or you get seg faults
    initModule_nonfree();
    
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
    if (!prob) {
        for (unsigned int it = 0; it < K; it++) {
            Mat codeBook;
            vector<vector<Mat>> imageDescriptors;
            int partitionStart = it * partition;
            TrainBoW(people_set, codeBook, imageDescriptors, numCodewords, partitionStart, partition);
            double recognition;
            TestBoW(people_set, codeBook, imageDescriptors, partitionStart, partition, recognition);
            avg_recognition += recognition;
        }
    } else {
        for (unsigned int it = 0; it < K; it++) {
            Mat codeBook;
            vector<vector<Mat>> imageDescriptors;
            int partitionStart = it * partition;
            vector<Mat> covar_mat;
            vector<Mat> mean_vectors;
            TrainBoWProb(people_set, codeBook, imageDescriptors, numCodewords, partitionStart, partition, covar_mat, mean_vectors);
            double recognition;
            TestBoWProb(people_set, codeBook, partitionStart, partition, covar_mat, mean_vectors, recognition);
            avg_recognition += recognition;
        }
    }
    avg_recognition /= K;

    cout << ((prob) ? "probabilistic: " : "regular: ") << numCodewords << ": " << avg_recognition * 100 << "%" << endl;
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

        for (unsigned int j = (partitionStart == 0) ? partitionSize : 0; j < people_set[i].size(); j++) {
            if (j == partitionStart) {
                j += partitionSize;
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

#if DISPLAY
    ofstream writefile;
    writefile.open("histograms.txt");
    int correct = 0;
    int incorrect = 0;
#endif

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
            unsigned int best_m = -1;
            unsigned int best_n = -1;
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
                        best_n = n;
                    }
                }
            }

            // assign the category index
            if (best_m == i) {
                numCorrect++;
#if DISPLAY
                if (correct < 2) {
                    writefile << "correct" << endl;
                    writefile << imageDescriptors[best_m][best_n] << endl;
                    writefile << "\n=========================\n" << endl;
                    correct++;

                    writefile << "histogram" << endl;
                    writefile << histogram << endl;
                    writefile << "\n========================\n" << endl;
                }
#endif
            }
#if DISPLAY
            else {
                if (incorrect < 2) {
                    writefile << "incorrect" << endl;
                    writefile << imageDescriptors[best_m][best_n] << endl;
                    writefile << "\n=========================\n" << endl;
                    incorrect++;

                    writefile << "histogram" << endl;
                    writefile << histogram << endl;
                    writefile << "\n========================\n" << endl;
                }
            }
#endif
        }
    }

    recognition = double(numCorrect) / double(people_set.size() * partitionSize);
}

void TrainBoWProb(vector< vector<Mat> > people_set,
                  Mat &codeBook,
                  vector<vector<Mat>> &imageDescriptors,
                  const int numCodewords,
                  const int partitionStart,
                  const int partitionSize,
                  vector<Mat> &covar_mat,
                  vector<Mat> &mean_vectors) {

    TrainBoW(people_set, codeBook, imageDescriptors, numCodewords, partitionStart, partitionSize);

    // calculate covar and mean for each person
    for (int it = 0; it < imageDescriptors.size(); it++) {
        Mat covar, mean_vector;

        // change the input so that calcCovarMatrix can use it
        int samples = imageDescriptors[it].size();
        Mat input[samples];
        for (int j = 0; j < samples; j++) {
            Mat element = imageDescriptors[it][j];
            element.convertTo(element, CV_64F);
            input[j] = element;
        }

        calcCovarMatrix(input, samples, covar, mean_vector, CV_COVAR_NORMAL);
        covar_mat.push_back(covar);
        mean_vectors.push_back(mean_vector);
    }
}

void TestBoWProb(vector< vector<Mat> > people_set,
                 const Mat codeBook,
                 const int partitionStart,
                 const int partitionSize,
                 const vector<Mat> covar,
                 const vector<Mat> mean_vector,
                 double &recognition) {

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
            double mle = 0;
            int best_match = -1;

            // detect key points
            vector<KeyPoint> keyPoints;
            featureDetector->detect(I, keyPoints);

            // Compute histogram representation
            Mat histogram;
            bowDExtractor->compute2(I, keyPoints, histogram);
            int dimension = histogram.cols;

            // calculate the maximum likelihood estimation
            for (unsigned int it = 0; it < covar.size(); it++) {
                histogram.convertTo(histogram, mean_vector[it].type());
                Mat diff_vector = histogram - mean_vector[it];
                Mat variance = diff_vector * covar[it].inv() * diff_vector.t();
                double likelihood = exp(-0.5 * sum(variance)[0]) / (pow(2 * pi, dimension / 2) * determinant(covar[it]));
                if (likelihood > mle) {
                    mle = likelihood;
                    best_match = it;
                }
            }
            if (best_match == i) numCorrect++;
        }
    }
    recognition = (double)numCorrect / (double)(people_set.size() * partitionSize);
}
