#ifndef HPset_h
#define HPset_h

using namespace cv;
using namespace std;

class HPset {
private:
    vector<vector<Mat>> images;

public:
    HPset(string path, bool UNIXenv = true);

    Mat get(int personId, int serie, int tilt, int pan);

    void getPersonSet(int personId, int serie, vector<Mat>& set);

    void getPoseSet(int tilt, int pan, vector<Mat>& set);

    Mat getAllImage(int personId, int serie);
};

#endif /* HPset_h */
