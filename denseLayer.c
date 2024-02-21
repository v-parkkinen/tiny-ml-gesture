#include "denseLayer.h"

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "weights.h"

// Function to initialize DenseLayer
void initialize_dense_layer(DenseLayer *layer, int input_size, int output_size) {
    layer->input_size = input_size;
    layer->output_size = output_size;

    layer->W = (double *) dense_layer_weights;
    layer->b = (double *) dense_layer_biases;
}

// Function to perform forward pass in DenseLayer
double* forward_dense_layer(DenseLayer *layer, double *x) {
    // Allocate memory for result
    double *result = (double *)malloc(layer->output_size * sizeof(double));

    // Calculate z
    for (int i = 0; i < layer->output_size; i++) {
        result[i] = 0.0;
        for (int j = 0; j < layer->input_size; j++) {
            result[i] += layer->W[i * layer->input_size + j] * x[j];
        }
        result[i] += layer->b[i];
    }

    // Calculate exp_z
    double max_z = result[0];
    for (int i = 1; i < layer->output_size; i++) {
        if (result[i] > max_z) {
            max_z = result[i];
        }
    }

    double exp_sum = 0.0;
    for (int i = 0; i < layer->output_size; i++) {
        result[i] = exp(result[i] - max_z);
        exp_sum += result[i];
    }

    // Normalize and return
    for (int i = 0; i < layer->output_size; i++) {
        result[i] /= exp_sum;
    }

    return result;
}
