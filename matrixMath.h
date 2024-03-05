
// matrixMath.h
#ifndef MATRIXMATH_H
#define MATRIXMATH_H

class MatrixMath {
public:
    static void vectorMatrixMultiply(const float *matrix, const float *vector, float *result, const int vectorSize, const int matrixSize);
    static void vectorMultiply(const float *vector1, const float *vector2, float *result, const int vectorSize);
    static void vectorAdd(const float *vector1, const float *vector2, float *result, const int vectorSize);
    static void sigmoid(const float *input, float *output, const int size);
    static void tanhActivation(const float *input, float *output, const int size);
};

#endif