#include "lstmCell.h"
#include "weights.c"

void initialize_lstm_cell(LSTMCell *cell, int input_size, int hidden_size) {
    cell->input_size = input_size;
    cell->hidden_size = hidden_size;

    cell->W_ix = (float *) lstm_ix_weights;
    cell->W_ih = (float *) lstm_ih_weights;
    cell->b_i = (float *) lstm_i_biases;

    cell->W_fx = (float *) lstm_fx_weights;
    cell->W_fh = (float *) lstm_fh_weights;
    cell->b_f = (float *) lstm_f_biases;

    cell->W_cx = (float *) lstm_cx_weights;
    cell->W_ch = (float *) lstm_ch_weights;
    cell->b_c = (float *) lstm_c_biases;

    cell->W_ox = (float *) lstm_ox_weights;
    cell->W_oh = (float *) lstm_oh_weights;
    cell->b_o = (float *) lstm_o_biases;

    // TODO: Initialize hidden state and cell state
    cell->h_t = (float *)malloc(hidden_size * sizeof(float));
    cell->c_t = (float *)malloc(hidden_size * sizeof(float));
}

void forward_lstm_cell(LSTMCell *cell, float *x_t) {
    // Input gate
    // TODO: this is incorrect
    float i_t[cell->hidden_size];
    for (int i = 0; i < cell->hidden_size; i++) {
        i_t[i] = sigmoid(cell->W_ix[i] * x_t[i] + cell->W_ih[i] * cell->h_t[i] + cell->b_i[i]);
    }

    // Forget gate
    float f_t[cell->hidden_size];
    for (int i = 0; i < cell->hidden_size; i++) {
        f_t[i] = sigmoid(cell->W_fx[i] * x_t[i] + cell->W_fh[i] * cell->h_t[i] + cell->b_f[i]);
    }

    // Cell state update
    float tilde_c_t[cell->hidden_size];
    for (int i = 0; i < cell->hidden_size; i++) {
        tilde_c_t[i] = tanh_activation(cell->W_cx[i] * x_t[i] + cell->W_ch[i] * cell->h_t[i] + cell->b_c[i]);
        cell->c_t[i] = f_t[i] * cell->c_t[i] + i_t[i] * tilde_c_t[i];
    }

    // Output gate
    float o_t[cell->hidden_size];
    for (int i = 0; i < cell->hidden_size; i++) {
        o_t[i] = sigmoid(cell->W_ox[i] * x_t[0] + cell->W_oh[i] * cell->h_t[i] + cell->b_o[i]);
    }

    // Hidden state update
    for (int i = 0; i < cell->hidden_size; i++) {
        cell->h_t[i] = o_t[i] * tanh_activation(cell->c_t[i]);
    }
}

