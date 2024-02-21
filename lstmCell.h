// lstmCell.h
#ifndef LSTMCELL_H
#define LSTMCELL_H


typedef struct {
    int input_size;
    int hidden_size;

    float *W_ix, *W_ih, *b_i;
    float *W_fx, *W_fh, *b_f;
    float *W_cx, *W_ch, *b_c;
    float *W_ox, *W_oh, *b_o;

    float *h_t, *c_t;
} LSTMCell;

void initialize_lstm_cell(LSTMCell *cell, int input_size, int hidden_size);
void forward_lstm_cell(LSTMCell *cell, float *x_t);
void free_lstm_cell(LSTMCell *cell);

#endif