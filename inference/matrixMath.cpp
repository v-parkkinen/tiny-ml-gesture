// matrixMath.cpp
#include <math.h>
#include <avr/pgmspace.h>

#include "matrixMath.h"

void MatrixMath::vectorMatrixMultiply(const float *matrix, const float *vector, float *result, const int vectorSize, const int matrixSize) {
    // NOTE: this function assumes that the matrix is stored in program memory!
    for (int i = 0; i < matrixSize; i++) {
        result[i] = 0;
        for (int j = 0; j < vectorSize; j++) {
            result[i] += vector[j] * pgm_read_float(&matrix[i * vectorSize + j]);
        }
    }
}

void MatrixMath::vectorMultiply(const float *vector1, const float *vector2, float *result, const int vectorSize) {
    for (int i = 0; i < vectorSize; i++) {
        result[i] = vector1[i] * vector2[i];
    }
}

void MatrixMath::vectorAdd(const float *vector1, const float *vector2, float *result, const int vectorSize) {
    for (int i = 0; i < vectorSize; i++) {
        result[i] = vector1[i] + vector2[i];
    }
}

void MatrixMath::sigmoid(const float *input, float *output, const int size) {
    for (int i = 0; i < size; i++) {
        output[i] = 1.0 / (1.0 + exp(-input[i]));
    }
}

void MatrixMath::tanhActivation(const float *input, float *output, const int size) {
    for (int i = 0; i < size; i++) {
        output[i] = tanh(input[i]);
    }
}