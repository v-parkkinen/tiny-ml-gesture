// weights.h
#ifndef WEIGHTS_H
#define WEIGHTS_H

#include <avr/pgmspace.h>


extern const float lstmIxWeights[60];
extern const float lstmIhWeights[100];
extern const float lstmFxWeights[60];
extern const float lstmFhWeights[100];
extern const float lstmCxWeights[60];
extern const float lstmChWeights[100];
extern const float lstmOxWeights[60];
extern const float lstmOhWeights[100];
extern const float lstmIBiases[10];
extern const float lstmFBiases[10];
extern const float lstmCBiases[10];
extern const float lstmOBiases[10];
extern const float denseLayerWeights[50];
extern const float denseLayerBiases[5];

#endif