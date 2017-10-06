//  Created by 施浩琪 on 2017/10/5.
//  Copyright © 2017年 施浩琪. All rights reserved.

#include "nn.hpp"
#include <iostream>

typedef double(*CurveFunc)(double);
TrainData generateData(CurveFunc fun, double min, double max, int numPoints)
{
    TrainData data;
    for (int i=0; i<numPoints; ++i) {
        double zeroToOne = i/double(numPoints);
        double x = zeroToOne*(max-min) + min;
        double y = fun(x);
        Sample sp;
        sp.x.push_back(x);
        sp.y.push_back(y);
        data.push_back(sp);
    }
    return data;
}

int main(void)
{
    // generate the training data
    TrainData data = generateData(sin, -3, 3, 30);
    
    // build a network
    NN network;
    vector<int> structure = {1,10,10,1};
    vector<ActivateFun> funs = {linear,sigmoid,sigmoid,linear};
    network.init(structure,funs);
    
    // train & output
    int n = 0;
    while (network.RMS > 0.03) {
        network.train(data);
        if ( (n++ % 100) == 0)
            cout << "RMS : " << network.RMS << endl;
    }
    cout << "Done." << endl;
    return 0;
}
