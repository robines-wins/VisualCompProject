#ifndef QMULset_h
#define QMULset_h

using namespace cv;
using namespace std;

class QMULset {
private:
    vector<vector<Mat>> img;
    map<string, int> nameMap;

    Mat getByIndex(int face, int tilt, int pan);

public:
    QMULset(string path,bool UNIXenv = true);

    Mat get(string subjectName, int tilt, int pan);

    Mat get(int subjectNameIndex, int tilt, int pan);

    void getPersonSet(String subjectName, vector<Mat>& set);

    void getPersonSet(int index, vector<Mat>& set);

    void getPoseSet(int tilt, int pan, vector<Mat>& set);

    void getCoarsePoseSet(vector<int> tiltClasses, vector<int> panClasses, int tiltIndex, int panIndex, vector<Mat>& set);

    void getAll(vector<Mat>& set);

    Mat getAllImage(string subjectName);

    Mat getAllImage(int subjectIndex);
};

#endif /* QMULset_h */
