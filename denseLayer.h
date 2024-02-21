// denseLayer.h
#ifndef DENSELAYER_H
#define DENSELAYER_H

typedef struct {
    int input_size;
    int output_size;
    double *W;
    double *b;
} DenseLayer;

void initialize_dense_layer(DenseLayer *layer, int input_size, int output_size);
double* forward_dense_layer(DenseLayer *layer, double *x);
void free_dense_layer(DenseLayer *layer);

#endif