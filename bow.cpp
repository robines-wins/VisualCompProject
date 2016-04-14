#include <string>
#include "bow.h"

#define DISPLAY 0
#define PEOPLE 31
#define PICTURES_PER_PERSON 133
#define TESTING 5

/* Train BoW */
void TrainBoW(QMULset Dataset, Mat &codeBook, vector<vector<Mat>> &imageDescriptors, const int numCodewords)
{
    // Create a SIFT feature detector object
    Ptr<FeatureDetector> featureDetector = FeatureDetector::create("SIFT");

    // Create a SIFT descriptor extractor object
    Ptr<DescriptorExtractor> descriptorExtractor = DescriptorExtractor::create("SIFT");

    // Mat object to store all the SIFT descriptors of all training categories
    Mat D;

    // loop for each person
    for (unsigned int i = 0; i < PEOPLE; i++) {
        vector<Mat> set;
        Dataset.getPersonSet(i, set);
        // each image of each category
        for (unsigned int j = 0; j < set.size()-TESTING; j++) {
            Mat I = set[j];

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
    for (unsigned int i = 0; i < PEOPLE; i++) {
        // each image of each category
        vector<Mat> category_descriptors;
        Mat average_mat = Mat::zeros(1, numCodewords, CV_32F);

        vector<Mat> set;
        Dataset.getPersonSet(i, set);

        for (unsigned int j = 0; j < set.size()-TESTING; j++) {
            Mat I = set[j];

            // detect key points
            vector<KeyPoint> keyPoints;
            featureDetector->detect(I, keyPoints);

            // Compute histogram representation and store in descriptors
            Mat histogram;
            bowDExtractor->compute2(I, keyPoints, histogram);
            category_descriptors.push_back(histogram);

            average_mat += histogram;
        }
        imageDescriptors.push_back(category_descriptors);

        average_mat = average_mat / set.size();
    }
}

/* Test BoW */
void TestBoW(QMULset Dataset, const Mat codeBook, const vector<vector<Mat>> imageDescriptors)
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
    for (unsigned int i = 0; i < PEOPLE; i++) {
        vector<Mat> set;
        Dataset.getPersonSet(i, set);

        // each image of each category
        for (unsigned int j = set.size() - TESTING; j < set.size(); j++) {
            Mat I = set[j];

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

    double ratio = double(numCorrect) / double(PEOPLE * TESTING);
    cout << ratio * 100 << "% recognition" << endl;
}

