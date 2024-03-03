#include <stdlib.h>

#include "lstmCell.h"
#include "weights.h"
#include "matrixMath.h"

LSTMCell::LSTMCell(int input_size, int hidden_size) {
    this->input_size = input_size;
    this->hidden_size = hidden_size;
    this->h_t = new float[hidden_size];
    this->c_t = new float[hidden_size];
    for (int i = 0; i < hidden_size; i++) {
        this->h_t[i] = 0.0;
        this->c_t[i] = 0.0;
    }
}

LSTMCell::~LSTMCell() {
    delete[] h_t;
    delete[] c_t;
}

void LSTMCell::forward(float x_t[6]) {
    // Input gate
    float i_t[this->hidden_size];
    float ig_1[this->hidden_size];
    float ig_2[this->hidden_size];

    MatrixMath::vectorMatrixMultiply(lstm_ix_weights, x_t, ig_1, this->input_size, this->hidden_size);
    MatrixMath::vectorMatrixMultiply(lstm_ih_weights, this->h_t, ig_2, this->hidden_size, this->hidden_size);
    MatrixMath::vectorAdd(ig_1, ig_2, i_t, this->hidden_size);
    MatrixMath::vectorAdd(i_t, lstm_i_biases, i_t, this->hidden_size);
    MatrixMath::sigmoid(i_t, i_t, this->hidden_size);

    // Forget gate
    float f_t[this->hidden_size];
    float fg_1[this->hidden_size];
    float fg_2[this->hidden_size];
    MatrixMath::vectorMatrixMultiply(lstm_fx_weights, x_t, fg_1, this->input_size, this->hidden_size);
    MatrixMath::vectorMatrixMultiply(lstm_fh_weights, this->h_t, fg_2, this->hidden_size, this->hidden_size);
    MatrixMath::vectorAdd(fg_1, fg_2, f_t, this->hidden_size);
    MatrixMath::vectorAdd(f_t, lstm_f_biases, f_t, this->hidden_size);
    MatrixMath::sigmoid(f_t, f_t, this->hidden_size);

    // Cell state update
    float tilde_c_t[this->hidden_size];
    float ct_1[this->hidden_size];
    float ct_2[this->hidden_size];
    MatrixMath::vectorMatrixMultiply(lstm_cx_weights, x_t, ct_1, this->input_size, this->hidden_size);
    MatrixMath::vectorMatrixMultiply(lstm_ch_weights, this->h_t, ct_2, this->hidden_size, this->hidden_size);
    MatrixMath::vectorAdd(ct_1, ct_2, tilde_c_t, this->hidden_size);
    MatrixMath::vectorAdd(tilde_c_t, lstm_c_biases, tilde_c_t, this->hidden_size);
    
    MatrixMath::tanh_activation(tilde_c_t, tilde_c_t, this->hidden_size);
    MatrixMath::vectorMultiply(f_t, this->c_t, ct_1, this->hidden_size);
    MatrixMath::vectorMultiply(i_t, tilde_c_t, ct_2, this->hidden_size);
    MatrixMath::vectorAdd(ct_1, ct_2, this->c_t, this->hidden_size);


    // Output gate
    float o_t[this->hidden_size];
    float og_1[this->hidden_size];
    float og_2[this->hidden_size];
    MatrixMath::vectorMatrixMultiply(lstm_ox_weights, x_t, og_1, this->input_size, this->hidden_size);
    MatrixMath::vectorMatrixMultiply(lstm_oh_weights, this->h_t, og_2, this->hidden_size, this->hidden_size);
    MatrixMath::vectorAdd(og_1, og_2, o_t, this->hidden_size);
    MatrixMath::vectorAdd(o_t, lstm_o_biases, o_t, this->hidden_size);
    MatrixMath::sigmoid(o_t, o_t, this->hidden_size);

    // Hidden state update
    float tanh_ct[this->hidden_size];
    MatrixMath::tanh_activation(this->c_t, tanh_ct, this->hidden_size);
    MatrixMath::vectorMultiply(o_t, tanh_ct, this->h_t, this->hidden_size);
}

