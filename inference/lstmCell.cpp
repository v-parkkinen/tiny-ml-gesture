#include "lstmCell.h"
#include "weights.h"
#include "matrixMath.h"

LSTMCell::LSTMCell(int inputSize, int hiddenSize) {
    this->inputSize = inputSize;
    this->hiddenSize = hiddenSize;
    this->hiddenState = new float[hiddenSize];
    this->cellState = new float[hiddenSize];
    for (int i = 0; i < hiddenSize; i++) {
        this->hiddenState[i] = 0.0;
        this->cellState[i] = 0.0;
    }
}

LSTMCell::~LSTMCell() {
    delete[] hiddenState;
    delete[] cellState;
}

void LSTMCell::forward(float x_t[6]) {
    // Input gate
    float inputGate[this->hiddenSize];
    float ig_1[this->hiddenSize];
    float ig_2[this->hiddenSize];

    MatrixMath::vectorMatrixMultiply(lstmIxWeights, x_t, ig_1, this->inputSize, this->hiddenSize);
    MatrixMath::vectorMatrixMultiply(lstmIhWeights, this->hiddenState, ig_2, this->hiddenSize, this->hiddenSize);
    MatrixMath::vectorAdd(ig_1, ig_2, inputGate, this->hiddenSize);
    MatrixMath::vectorAdd(inputGate, lstmIBiases, inputGate, this->hiddenSize);
    MatrixMath::sigmoid(inputGate, inputGate, this->hiddenSize);

    // Forget gate
    float forgetGate[this->hiddenSize];
    float fg_1[this->hiddenSize];
    float fg_2[this->hiddenSize];
    MatrixMath::vectorMatrixMultiply(lstmFxWeights, x_t, fg_1, this->inputSize, this->hiddenSize);
    MatrixMath::vectorMatrixMultiply(lstmFhWeights, this->hiddenState, fg_2, this->hiddenSize, this->hiddenSize);
    MatrixMath::vectorAdd(fg_1, fg_2, forgetGate, this->hiddenSize);
    MatrixMath::vectorAdd(forgetGate, lstmFBiases, forgetGate, this->hiddenSize);
    MatrixMath::sigmoid(forgetGate, forgetGate, this->hiddenSize);

    // Cell state update
    float tilde_c_t[this->hiddenSize];
    float ct_1[this->hiddenSize];
    float ct_2[this->hiddenSize];
    MatrixMath::vectorMatrixMultiply(lstmCxWeights, x_t, ct_1, this->inputSize, this->hiddenSize);
    MatrixMath::vectorMatrixMultiply(lstmChWeights, this->hiddenState, ct_2, this->hiddenSize, this->hiddenSize);
    MatrixMath::vectorAdd(ct_1, ct_2, tilde_c_t, this->hiddenSize);
    MatrixMath::vectorAdd(tilde_c_t, lstmCBiases, tilde_c_t, this->hiddenSize);
    
    MatrixMath::tanhActivation(tilde_c_t, tilde_c_t, this->hiddenSize);
    MatrixMath::vectorMultiply(forgetGate, this->cellState, ct_1, this->hiddenSize);
    MatrixMath::vectorMultiply(inputGate, tilde_c_t, ct_2, this->hiddenSize);
    MatrixMath::vectorAdd(ct_1, ct_2, this->cellState, this->hiddenSize);


    // Output gate
    float outputGate[this->hiddenSize];
    float og_1[this->hiddenSize];
    float og_2[this->hiddenSize];
    MatrixMath::vectorMatrixMultiply(lstmOxWeights, x_t, og_1, this->inputSize, this->hiddenSize);
    MatrixMath::vectorMatrixMultiply(lstmOhWeights, this->hiddenState, og_2, this->hiddenSize, this->hiddenSize);
    MatrixMath::vectorAdd(og_1, og_2, outputGate, this->hiddenSize);
    MatrixMath::vectorAdd(outputGate, lstmOBiases, outputGate, this->hiddenSize);
    MatrixMath::sigmoid(outputGate, outputGate, this->hiddenSize);

    // Hidden state update
    float tanh_ct[this->hiddenSize];
    MatrixMath::tanhActivation(this->cellState, tanh_ct, this->hiddenSize);
    MatrixMath::vectorMultiply(outputGate, tanh_ct, this->hiddenState, this->hiddenSize);
}

