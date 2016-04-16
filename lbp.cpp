#include "lbp.h"

#define DISPLAY 0
#define PICTURES_PER_PERSON 133
#define TESTING 5

using namespace cv;
using namespace std;

Mat createLbpHistorgram(Mat image, int numGridHorizontal, int numGridVertical) {
	// Table for bins to put each lbp number
	const char binTable[256] = {
	1, 2, 3, 4, 5, 0, 6, 7, 8, 0, 0, 0, 9, 0, 10, 11,
	12, 0, 0, 0, 0, 0, 0, 0, 13, 0, 0, 0, 14, 0, 15, 16,
	17, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
	18, 0, 0, 0, 0, 0, 0, 0, 19, 0, 0, 0, 20, 0, 21, 22,
	23, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
	0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
	24, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
	25, 0, 0, 0, 0, 0, 0, 0, 26, 0, 0, 0, 27, 0, 28, 29,
	30, 31, 0, 32, 0, 0, 0, 33, 0, 0, 0, 0, 0, 0, 0, 34,
	0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 35,
	0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
	0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 36,
	37, 38, 0, 39, 0, 0, 0, 40, 0, 0, 0, 0, 0, 0, 0, 41,
	0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 42,
	43, 44, 0, 45, 0, 0, 0, 46, 0, 0, 0, 0, 0, 0, 0, 47,
	48, 49, 0, 50, 0, 0, 0, 51, 52, 53, 0, 54, 55, 56, 57,58};

	cv::Mat lbpImage(image.rows, image.cols, CV_8UC1);
    int imageWidth = image.cols/numGridHorizontal;
    int imageHeight = image.rows/numGridVertical;
    Mat outputHistogram = Mat::zeros(0, 0, CV_32F);

	int center = 0;
	int center_lbp = 0;
	int radius = 2;
	int a = 0; //Not usefull but it count numberof iteration

    // Iterate through each grid
    for(int i = 0; i < numGridVertical; i++) {
        for(int j = 0; j < numGridHorizontal; j++) {
            Mat imagePatch(image, Range(i*imageHeight,(i+1)*imageHeight), Range(j*imageWidth,(j+1)*imageWidth));
            Mat hist = Mat::zeros(1,59,CV_32F);
            Mat histNorm = Mat::zeros(1,59,CV_32F);

            //Iterate through each pixel in the current grid and sets the lbp for each pixel
            for(int row = radius; row < imagePatch.rows - radius; row++){
            	for(int col = radius; col < imagePatch.cols - radius; col++){

			center = imagePatch.at<int>(row, col);
			center_lbp = 0;

			if (center <= imagePatch.at<int>(row - radius, col - radius)){
				center_lbp += 1;
            }

			if (center <= imagePatch.at<int>(row - radius, col)){
				center_lbp += 2;
            }

			if (center <= imagePatch.at<int>(row - radius, col + radius)){
				center_lbp += 4;
            }

			if (center <= imagePatch.at<int>(row, col - radius)){
				center_lbp += 8;
            }

			if (center <= imagePatch.at<int>(row, col + radius)){
				center_lbp += 16;
            }

			if (center <= imagePatch.at<int>(row + radius, col - radius)){
				center_lbp += 32;
            }

			if (center <= imagePatch.at<int>(row + radius, col)){
				center_lbp += 64;
            }
			if (center <= imagePatch.at<int>(row + radius, col + radius)){
				center_lbp += 128;
            }


			//imagePatch.at<int>(row, col) = center_lbp;
			// Calculate histogram
			// This creates a unifrom 1X59 matrix (filters out the transition)
			hist.at<float>(0,binTable[center_lbp]) += 1;
			a = a+1; // Not usefull only for me to count the iteration

			//Normalize each histogram
			cv::normalize(hist, histNorm, NORM_L2);
            	}
            }
            outputHistogram.push_back(histNorm);
        }
    }
    //Reshape histogram into 1D
    //outputHistogram = outputHistogram.reshape(0,1);
    return outputHistogram;
}

// Function that creates the spatial pyramid histogram
Mat createSpatialPyramidHistogram(Mat image, int Level) {
    Mat histTemp =Mat::zeros(0, 0, CV_32F);
    Mat histResult= Mat::zeros(0, 0, CV_32F);

	for(int l = 0; l <= Level; l++){
		int gridX = pow(2,l);
		int gridY = pow(2,l);
		histTemp = createLbpHistorgram(image, gridX ,gridY);
		histResult.push_back(histTemp);
	}

	histResult = histResult.reshape(0,1);
	return histResult;
}

// Function that calculates the Chi Square of 2 histograms
double calculateChiSquare(Mat histogram1, Mat histogram2) {
	Mat difference = histogram1 - histogram2;
	Mat numerator = difference.mul(difference);
	Mat denominator = 1/(histogram1 + histogram2);
	double chiDistance = sum(numerator.mul(denominator))[0];

	return chiDistance;
}

//Returns the spatial level of the histogram
double spatialLevelOf(Mat histogram){
	int cols = histogram.cols;
	int level = -1;

	if(cols == 59){
		level = 0;
	}
	if(cols == 295){
		level = 1;
	}
	if(cols == 1239){
		level = 2;
	}
	if(cols == 5015){
		level = 3;
	}
	return level;
}

//Calculate the Spatial Chi of 2 histograms
double calculateSpatialChi(Mat histogram1, Mat histogram2){
	int level = spatialLevelOf(histogram1);
	double spatialChi;
	double chi0 = 0;
	double chi1 = 0;
	double chi2 = 0;
	double chi3 = 0;

	if(level >= 0){
		Mat hist1ChiLevel0 = histogram1(Range::all(), Range(0,59));
		Mat hist2ChiLevel0 = histogram2(Range::all(), Range(0,59));
		chi0 = calculateChiSquare(hist1ChiLevel0, hist2ChiLevel0);
	}
	if(level >= 1){
		Mat hist1ChiLevel1 = histogram1(Range::all(), Range(0,295));
		Mat hist2ChiLevel1 = histogram2(Range::all(), Range(0,295));
		chi1 = calculateChiSquare(hist1ChiLevel1, hist2ChiLevel1);
	}
	if(level >= 2){
		Mat hist1ChiLevel2 = histogram1(Range::all(), Range(0,1239));
		Mat hist2ChiLevel2 = histogram2(Range::all(), Range(0,1239));
		chi2 = calculateChiSquare(hist1ChiLevel2, hist2ChiLevel2);
	}
	if(level >= 3){
		Mat hist1ChiLevel3 = histogram1(Range::all(), Range(0,5015));
		Mat hist2ChiLevel3 = histogram2(Range::all(), Range(0,5015));
		chi3 = calculateChiSquare(hist1ChiLevel3, hist2ChiLevel3);
	}

	spatialChi = chi0/pow(2,level) + chi1/pow(2,level) + chi2/pow(2,level-1) + chi3/pow(2,level-2);
	return spatialChi;
}

void TrainLbp(vector< vector<Mat> >& people_set,
        vector<vector <Mat> > &imageDescriptors,
        const int levels,
        const int partitionStart,
        const int partitionSize)
{

    for (unsigned int i = 0; i < people_set.size(); i++) {
        // each image of each category
        vector<Mat> category_descriptors;
        for (unsigned int j = (partitionStart == 0) ? partitionSize : 0; j < people_set[i].size(); j++) {
            if (j == partitionStart) {
                j += partitionSize-1;
                continue;
            }

            Mat I = people_set[i][j];

            // Compute histogram representation and store in descriptors
            Mat histogram = createSpatialPyramidHistogram(I, levels);

            category_descriptors.push_back(histogram);
        }
        imageDescriptors.push_back(category_descriptors);
    }
}

void TestLbp(vector< vector<Mat> >& people_set,
             const vector<vector <Mat> >& imageDescriptors,
             const int levels,
             const int partitionStart,
             const int partitionSize,
             double &recognition)
{
	int numCorrect = 0;
	Mat D;

    // loop for each category
    for (unsigned int i = 0; i < people_set.size(); i++) {
        // each image of each category
        for (unsigned int j = partitionStart; j < partitionStart+partitionSize; j++) {
            Mat I = people_set[i][j];

            // Compute histogram representation
            Mat histogram = createSpatialPyramidHistogram(I, levels);

            // compare and find the best matching histogram
            double best_dist = numeric_limits<double>::max();
            unsigned int best_m = 0;
            for (unsigned int m = 0; m < imageDescriptors.size(); m++) {
                for (unsigned int n = 0; n < imageDescriptors[m].size(); n++) {
                    double dist = calculateSpatialChi(histogram, imageDescriptors[m][n]);
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

    recognition = double(numCorrect) / double(people_set.size() * partitionSize);
}

void TrainLbpProb(vector< vector<Mat> >& people_set,
                  vector<vector<Mat> > &imageDescriptors,
                  const int levels,
                  const int partitionStart,
                  const int partitionSize,
                  vector<Mat> &covar_mat,
                  vector<Mat> &mean_vectors) {

    TrainLbp(people_set, imageDescriptors, levels, partitionStart, partitionSize);

    // calculate covar and mean for each person
    for (int it = 0; it < imageDescriptors.size(); it++) {
        Mat covar, mean_vector;

        // change the input so that calcCovarMatrix can use it
        int samples = imageDescriptors[it].size();
        Mat *input = new Mat[samples];
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

void TestLbpProb(vector< vector<Mat> >& people_set,
                 const int levels,
                 const int partitionStart,
                 const int partitionSize,
                 const vector<Mat>& covar,
                 const vector<Mat>& mean_vector,
                 double &recognition) {

    int numCorrect = 0;

    // loop for each category
    for (unsigned int i = 0; i < people_set.size(); i++) {
        // each image of each category
        for (unsigned int j = partitionStart; j < partitionStart+partitionSize; j++) {
            Mat I = people_set[i][j];
            double mle = 0;
            int best_match = -1;

            Mat histogram = createSpatialPyramidHistogram(I, levels);
            int dimension = histogram.cols;

            // calculate the maximum likelihood estimation
            for (unsigned int it = 0; it < covar.size(); it++) {
                histogram.convertTo(histogram, mean_vector[it].type());
                Mat diff_vector = histogram - mean_vector[it];
                Mat variance = diff_vector * covar[it].inv() * diff_vector.t();
                double likelihood = exp(-0.5 * sum(variance)[0]) / (pow(2 * pi, dimension / 2) * sqrt(norm(covar[it])));
                if (likelihood > mle) {
                    mle = likelihood;
                    best_match = it;
                }
            }
            if (best_match == i) numCorrect++;
        }
    }
    recognition = numCorrect / (people_set.size() * partitionSize);
}

double LbpkFoldsCrossValidation(vector< vector<Mat> >& people_set, const int levels, int k, bool prob) {
    // randomly shuffle the input data set
    srand(time(0));
	for (size_t it = 0; it < people_set.size(); it++) {
		random_shuffle(people_set[it].begin(), people_set[it].end());
	}

    // validate for k = 7 fold
    int partition = PICTURES_PER_PERSON / k;
    double avg_recognition = 0;
    if (!prob) {
        for (unsigned int it = 0; it < k; it++) {
            vector<vector<Mat> > imageDescriptors;
            int partitionStart = it * partition;
            TrainLbp(people_set, imageDescriptors, levels, partitionStart, partition);
            double recognition;
            TestLbp(people_set, imageDescriptors, levels, partitionStart, partition, recognition);
            avg_recognition += recognition;
        }
    } else {
        for (unsigned int it = 0; it < k; it++) {
            Mat codeBook;
            vector<vector<Mat> > imageDescriptors;
            int partitionStart = it * partition;
            vector<Mat> covar_mat;
            vector<Mat> mean_vectors;
            TrainLbpProb(people_set, imageDescriptors, levels, partitionStart, partition, covar_mat, mean_vectors);
            double recognition;
            TestLbpProb(people_set, levels, partitionStart, partition, covar_mat, mean_vectors, recognition);
            avg_recognition += recognition;
        }
    }
    return avg_recognition / k;
}
