#include <iostream>
#include <cmath>

#include "denseLayer.h"
#include "weights.h"
#include "matrixMath.h"

DenseLayer::DenseLayer(int input_size, int output_size) {
    this->input_size = input_size;
    this->output_size = output_size;
}

DenseLayer::~DenseLayer() {
}

float* DenseLayer::forward(float *x, float *result) {
    // Calculate z
    float z[5];
    MatrixMath::vectorMatrixMultiply(dense_layer_weights, x, z, 10, 5);
    MatrixMath::vectorAdd(z, dense_layer_biases, z, 5);

    // Calculate exp_z
    float max_z = z[0];
    for (int i = 1; i < output_size; i++) {
        if (z[i] > max_z) {
            max_z = z[i];
        }
    }

    float exp_sum = 0.0;
    for (int i = 0; i < output_size; i++) {
        result[i] = std::exp(z[i] - max_z);
        exp_sum += result[i];
    }

    // Normalize and return
    for (int i = 0; i < output_size; i++) {
        result[i] /= exp_sum;
    }

    return result;
}