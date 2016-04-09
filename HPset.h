#ifndef HPset_h
#define HPset_h

using namespace cv;
using namespace std;

class HPset {
private:
    vector<vector<Mat>> images;

public:
    HPset(string path, bool UNIXenv = true);
};

#endif /* HPset_h */
