///  Created by 施浩琪 on 2017/10/5.
//  Copyright © 2017年 施浩琪. All rights reserved.

#ifndef data_h
#define data_h
#include <vector>
using namespace std;

typedef vector<double> Vec;
typedef Vec InputVec;
typedef Vec OutputVec;

struct Sample {
    Vec x;
    Vec y;
};

typedef vector<Sample> TrainData;

#endif /* data_h */
