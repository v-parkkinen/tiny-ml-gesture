#ifndef DENSELAYER_H
#define DENSELAYER_H

class DenseLayer {
public:
    int inputSize;
    int outputSize;


    DenseLayer(int inputSize, int output_size);
    ~DenseLayer();

    float* forward(float *input, float *output);
};

#endif