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

    vector<Mat> getSet(String subjectName);

    vector<Mat> getSet(int tilt, int pan);

    vector<Mat> getSet(int index);

    vector<Mat> getAll();

    Mat allImageFromSubject(string subjectName);

    Mat allImageFromSubject(int subjectIndex);

};

#endif /* QMULset_h */
