//
//  QMULset.h
//  Project
//
//  Created by Robin Solignac on 29/03/2016.
//  Copyright © 2016 Robin Solignac. All rights reserved.
//

#ifndef QMULset_h
#define QMULset_h

#include <stdio.h>

#define QMUL_TILT_COUNT 19
#define QMUL_PAN_COUNT 7
#define QMUL_IMG_SIZE 100

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
