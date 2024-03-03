#ifndef DENSELAYER_H
#define DENSELAYER_H

class DenseLayer {
public:
    int input_size;
    int output_size;


    DenseLayer(int input_size, int output_size);
    ~DenseLayer();

    float* forward(float *x, float *output);
};

#endif