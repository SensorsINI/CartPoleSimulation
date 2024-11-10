// network.c

#include <stdio.h>
#include <math.h>
#include <stdlib.h>

// Include the weights and biases stored as arrays in separate .c file
#include "network.h"
#include "network_parameters.c"

void applyTanh(float* x, int size) {
    for (int i = 0; i < size; i++) {
        x[i] = tanhf(x[i]);  // Apply tanh activation function
    }
}

void matMul(const float* matrix, const float* vec, float* result, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        result[i] = 0;
        for (int j = 0; j < cols; j++) {
            result[i] += matrix[i * cols + j] * vec[j];  // Matrix multiplication (matrix * vector)
        }
    }
}

void addBias(const float* bias, float* vec, int size) {
    for (int i = 0; i < size; i++) {
        vec[i] += bias[i];  // Adding bias to each element in the vector
    }
}

// Updated forwardPass to handle three Dense layers
void C_Network_Evaluate(float* inputs, float* outputs) {

    // First layer (input to layer 1)
    float* layer1 = (float*) malloc(LAYER1_SIZE * sizeof(float));
    matMul(weights1, inputs, layer1, LAYER1_SIZE, INPUT_SIZE);
    addBias(bias1, layer1, LAYER1_SIZE);
    applyTanh(layer1, LAYER1_SIZE);  // tanh activation

    // Second layer (layer 1 to layer 2)
    float* layer2 = (float*) malloc(LAYER2_SIZE * sizeof(float));
    matMul(weights2, layer1, layer2, LAYER2_SIZE, LAYER1_SIZE);
    addBias(bias2, layer2, LAYER2_SIZE);
    applyTanh(layer2, LAYER2_SIZE);  // tanh activation

    // Third layer (layer 2 to output)
    matMul(weights3, layer2, outputs, LAYER3_SIZE, LAYER2_SIZE);
    addBias(bias3, outputs, LAYER3_SIZE);

    // Free dynamically allocated memory
    free(layer1);
    free(layer2);
}

