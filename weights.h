// weights.h
#ifndef WEIGHTS_H
#define WEIGHTS_H

extern const float lstm_ix_weights[60];
extern const float lstm_ih_weights[100];
extern const float lstm_fx_weights[60];
extern const float lstm_fh_weights[100];
extern const float lstm_cx_weights[60];
extern const float lstm_ch_weights[100];
extern const float lstm_ox_weights[60];
extern const float lstm_oh_weights[100];
extern const float lstm_i_biases[10];
extern const float lstm_f_biases[10];
extern const float lstm_c_biases[10];
extern const float lstm_o_biases[10];
extern const float dense_layer_weights[50];
extern const float dense_layer_biases[5];

#endif