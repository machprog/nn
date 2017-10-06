///  Created by 施浩琪 on 2017/10/5.
//  Copyright © 2017年 施浩琪. All rights reserved.

#ifndef nn_hpp
#define nn_hpp
#include <vector>
#include <iostream>
#include <assert.h>
#include <math.h>
#include <limits>
#include <time.h>
#include "data.h"

using namespace std;

typedef double(*ActivateFun)(double);
typedef double(*DActivateFun)(double);
typedef vector<vector<double>> Mat;

double sigmoid(double x);
double d_sigmoid(double x);
double linear(double x);
double d_linear(double x);

class NNLayer
{
private:
    int numNeuron;
    int numNeuronNext;
    Mat w;
    Vec b;
    NNLayer* nextLayer;
    NNLayer* previousLayer;
    
public:
    Vec x;
    Vec y;
    Vec delta;
    ActivateFun f;
    DActivateFun Df;
    void init(int n, int m, ActivateFun fun = sigmoid);
    void load(const InputVec& invec);
    void forwardToNextLayer();
    void backToPreviousLayer();
    void linkTo(NNLayer* pl, NNLayer* nl);
    void gradientDescent();
    void updateOutput();
};

class NN
{
private:
    int numLayer;
    vector<NNLayer> layers;
    
private:
    void forwardProp();
    void backProp();
    void loadDataToFirstLayer(const InputVec& invec);
    void updateWeights();
    void createLayerConnections();

public:
    double RMS;
    int numInteration;
    void init(vector<int>& numNeuronOfLayers, vector<ActivateFun> funs);
    void train(TrainData& data);
    OutputVec test(InputVec& x);
};

#endif /* nn_hpp */
