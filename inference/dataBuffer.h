#ifndef DATABUFFER_H
#define DATABUFFER_H

class DataBuffer {
private:
    static const int STACK_SIZE = 10;
    static const int ARRAY_SIZE = 6;

    float stack[STACK_SIZE][ARRAY_SIZE];
    int top;

public:
    DataBuffer();

    void push(const float data[ARRAY_SIZE]);
    float* access(int index);
};

#endif // DATABUFFER_H