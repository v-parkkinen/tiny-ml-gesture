#include "lstmCell.h"
#include "denseLayer.h"

typedef struct {
    int input_size;
    int hidden_size;
    LSTMCell *cell;
} LSTM;

void initialize_custom_lstm(LSTM *lstm, int input_size, int hidden_size) {
    lstm->input_size = input_size;
    lstm->hidden_size = hidden_size;

    lstm->cell = (LSTMCell *)malloc(sizeof(LSTMCell));
    initialize_lstm_cell(lstm->cell, input_size, hidden_size);
}

void forward_lstm(LSTM *lstm, double *x) {
    int seq_len = 10;  // Assuming one time step
    for (int t = 0; t < seq_len; t++) {
        forward_lstm_cell(lstm->cell, x);
    }
}

void free_lstm(LSTM *lstm) {
    free_lstm_cell(lstm->cell);
    free(lstm->cell);
}

int main() {
    int input_size =  10;
    int hidden_size =  6;
    int num_classes = 5;

    LSTM lstm;
    initialize_custom_lstm(&lstm, input_size, hidden_size);

    DenseLayer denseLayer;
    initialize_dense_layer(&denseLayer, input_size, num_classes);


    // Your input sequence (assuming one time step for simplicity)
    double x[input_size];
    
    // Forward pass
    forward_lstm(&lstm, x);
    forward_dense_layer(&denseLayer, x);

    // Cleanup
    free_lstm(&lstm);
    free_dense_layer(&denseLayer);
    
    return 0;
}