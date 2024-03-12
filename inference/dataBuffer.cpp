#include "dataBuffer.h"

DataBuffer::DataBuffer() : top(-1) {
    // Initialize all elements in the stack to 0
    for (int i = 0; i < STACK_SIZE; ++i) {
        for (int j = 0; j < ARRAY_SIZE; ++j) {
            stack[i][j] = 0.0f;
        }
    }
}

void DataBuffer::push(const float data[ARRAY_SIZE]) {
    top = (top + 1) % STACK_SIZE; // Move the top index circularly
    // Copy the new data into the stack at the top index
    for (int i = 0; i < ARRAY_SIZE; ++i) {
        stack[top][i] = data[i];
    }
}

float* DataBuffer::access(int index) {
    if (index < 0 || index >= STACK_SIZE) {
        return nullptr;
    }
    int requestedIndex = (top + index + 1) % STACK_SIZE;
    return stack[requestedIndex];
}