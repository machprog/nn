///  Created by 施浩琪 on 2017/10/5.
//  Copyright © 2017年 施浩琪. All rights reserved.

#include "nn.hpp"
#define ALPHA 0.03

DActivateFun D(ActivateFun f)
{
    return (f == sigmoid)? d_sigmoid : d_linear;
}

Vec matrixMul(Mat& A,Vec& x)
{
    assert(A[0].size()==x.size());
    
    Vec y;
    for (int i=0; i<A.size(); ++i) {
        double ele = 0.0;
        for (int j=0; j<x.size(); ++j) {
            ele += A[i][j] * x[j];
        }
        y.push_back(ele);
    }
    return y;
}

Vec matrixTMul(Mat& A,Vec& x)
{
    assert(A.size()==x.size());
    
    Vec y;
    for (int i=0; i<A[0].size(); ++i) {
        double ele = 0.0;
        for (int j=0; j<x.size(); ++j) {
            ele += A[j][i] * x[j];
        }
        y.push_back(ele);
    }
    return y;
}

Vec vecAdd(Vec& a,Vec& b)
{
    assert(a.size()==b.size());
    
    Vec y;
    for (size_t i=0; i<a.size(); ++i) {
        y.push_back(a[i]+b[i]);
    }
    return y;
}

double randf(double min,double max)
{
    double zeroToOne = double((rand()%101)/100.0);
    return zeroToOne*(max-min) + min;
}

double sigmoid(double x)
{
    return 1/(1+exp(-x));
}

double d_sigmoid(double x)
{
    return x * (1-x);
}

double linear(double x)
{
    return x;
}

double d_linear(double x)
{
    return 1;
}

double calcSquareError(Vec& a, Vec& b)
{
    double error = 0.0;
    for (size_t i=0; i<a.size(); ++i) {
        error += (a[i]-b[i])*(a[i]-b[i]);
    }
    return error;
}

void NN::init(vector<int> &numNeuronOfLayers, vector<ActivateFun> funs) {
    numLayer = (int)numNeuronOfLayers.size();
    RMS = numeric_limits<double>::max();
    numInteration = 0;
    
    for (int i=0; i<numLayer; ++i) {
        NNLayer layer;
        if( i != numLayer -1)
            layer.init(numNeuronOfLayers[i], numNeuronOfLayers[i+1], funs[i]); // init the first n-1 layers
        else
            layer.init(numNeuronOfLayers.back(), 0, funs[i]);//init the last layer
        layers.push_back(layer);
    }
    
    createLayerConnections();
}

void NN::train(TrainData &data) {
    assert(layers.back().y.size()==data[0].y.size());
    
    RMS=0.0;
    for (auto iter = data.begin(); iter!= data.end(); iter++) {
        Sample sp = (*iter);
        loadDataToFirstLayer(sp.x);
        forwardProp();
        Vec y = layers.back().y;
        RMS += calcSquareError(sp.y, y);
        for (size_t i=0; i<y.size(); i++) {
            layers.back().delta[i] = (y[i] - sp.y[i]) * layers.back().Df(y[i]);
        }
        backProp();
        updateWeights();
        numInteration += 1;
    }
    RMS = sqrt(RMS/data.size());
}

OutputVec NN::test(InputVec &x) {
    loadDataToFirstLayer(x);
    forwardProp();
    return layers.back().y;
}

void NN::forwardProp() { 
    for (auto iter = layers.begin(); iter != layers.end(); ++iter) {
        (*iter).forwardToNextLayer();
    };
}

void NN::loadDataToFirstLayer(const InputVec &invec) { 
    layers[0].load(invec);
}

void NN::backProp() {
    for (auto iter = layers.end()-1; iter != layers.begin(); --iter) {
        (*iter).backToPreviousLayer();
    };
}

void NN::updateWeights() { 
    for (auto iter = layers.end()-1; iter != layers.begin(); --iter) {
        (*iter).gradientDescent();
    };
}

void NN::createLayerConnections() {
    for (size_t i=0; i<layers.size(); ++i) {
        if( i == 0)
            layers[i].linkTo(nullptr, &layers[i+1]);
        else if(i < layers.size()-1)
            layers[i].linkTo(&layers[i-1], &layers[i+1]);
        else
            layers[i].linkTo(&layers[i-1], nullptr);
    };
}


void NNLayer::init(int n, int m, ActivateFun fun) {
    numNeuron = n;
    numNeuronNext = m;
    f = fun;
    Df = D(f);
    
    // init weights
    if (numNeuronNext == 0) // this is an output layer
    {
        //it hasn't weights, so do nothing
    }
    else
    {   //random init weights
        srand((unsigned int)time(NULL));
        w.resize(numNeuronNext);
        for (int i=0; i<numNeuronNext; ++i) {
            w[i].resize(numNeuron);
            for (int j=0; j<numNeuron; ++j) {
                w[i][j] = randf(-2.0, 2.0);
            }
        }
        //random init bias
        b.resize(numNeuronNext);
        for (int i=0; i<b.size(); ++i) {
            b[i] = randf(-2.0, -2.0);
        }
    }
    
    // init delta,y
    delta.resize(numNeuron);
    y.resize(numNeuron);
    x.resize(numNeuron);
}

void NNLayer::forwardToNextLayer() { 
    if (numNeuronNext == 0) // this is an output layer
    {
        updateOutput();  // needn't to prop to next layer
    }
    else
    {
        updateOutput();
        Vec x = matrixMul(w,y);
        x = vecAdd(x, b);
        nextLayer->load(x);
    }
}

void NNLayer::linkTo(NNLayer *pl, NNLayer *nl) {
    previousLayer = pl;
    nextLayer = nl;
}

void NNLayer::load(const InputVec &invec) {
    assert(invec.size()==x.size());
    x = invec;
}

void NNLayer::backToPreviousLayer() {
    Vec pDelta = matrixTMul(previousLayer->w,delta);
    Vec pY = previousLayer->y;
    DActivateFun Df = previousLayer->Df;
    for (size_t i=0; i < previousLayer->y.size(); i++) {
        previousLayer->delta[i] = pDelta[i] * Df(pY[i]);
    }
}

void NNLayer::gradientDescent() { 
    // update weights
    Vec y = previousLayer->y;
    for (size_t i=0; i<delta.size(); ++i) {
        for (size_t j=0; j<y.size(); ++j) {
            previousLayer->w[i][j] -= ALPHA * delta[i]*y[j];
        }
    }
    
    //update bias
    for (size_t i=0; i<delta.size(); ++i) {
        previousLayer->b[i] -= ALPHA * delta[i];
    }
}

void NNLayer::updateOutput() {
    for (size_t i=0; i<numNeuron; ++i) {
        y[i] = f(x[i]);
    }
}
