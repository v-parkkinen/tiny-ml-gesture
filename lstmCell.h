// lstmCell.h
#ifndef LSTMCELL_H
#define LSTMCELL_H

class LSTMCell {
public:
    int input_size;
    int hidden_size;
    float *h_t, *c_t;

    LSTMCell(int input_size, int hidden_size);
    ~LSTMCell();
    void forward(float *x_t);
};

#endif