// lstmCell.h
#ifndef LSTMCELL_H
#define LSTMCELL_H

class LSTMCell {
public:
    int inputSize;
    int hiddenSize;
    float *hiddenState, *cellState;

    LSTMCell(int inputSize, int hiddenSize);
    ~LSTMCell();
    void forward(float *input);
};

#endif