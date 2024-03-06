#include <math.h>

#include "denseLayer.h"
#include "weights.h"
#include "matrixMath.h"

DenseLayer::DenseLayer(int inputSize, int outputSize) {
    this->inputSize = inputSize;
    this->outputSize = outputSize;
}

DenseLayer::~DenseLayer() {
}

float* DenseLayer::forward(float *x, float *result) {
    // Calculate z
    float z[5];
    MatrixMath::vectorMatrixMultiply(denseLayerWeights, x, z, 10, 5);
    MatrixMath::vectorAdd(z, denseLayerBiases, z, 5);

    // Calculate exp_z
    float max_z = z[0];
    for (int i = 1; i < this->outputSize; i++) {
        if (z[i] > max_z) {
            max_z = z[i];
        }
    }

    float exp_sum = 0.0;
    for (int i = 0; i < this->outputSize; i++) {
        result[i] = exp(z[i] - max_z);
        exp_sum += result[i];
    }

    // Normalize and return
    for (int i = 0; i < this->outputSize; i++) {
        result[i] /= exp_sum;
    }

    return result;
}