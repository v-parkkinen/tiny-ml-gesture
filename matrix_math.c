void vectorMatrixMultiply(int vector[], int matrix[][3], int result[], int vectorSize, int matrixSize) {
    for (int i = 0; i < matrixSize; i++) {
        result[i] = 0;
        for (int j = 0; j < vectorSize; j++) {
            result[i] += vector[j] * matrix[j][i];
   
        } 
    }
}

