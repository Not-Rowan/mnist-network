// define neuralNetwork.h and include only once
#ifndef NEURALNETWORK_H
#define NEURALNETWORK_H

// By: Rowan Rothe

// Declare network structures

struct Layer {
    float **incomingWeights;    // weights from previous layer to this layer
    float *biases;              // biases for each neuron in this layer
    float *values;              // values of each neuron in this layer
    float *gradients;           // gradients of each neuron in this layer
    int neuronCount;            // number of neurons in this layer
};
typedef struct Layer Layer;

struct Network {
    Layer *layers;              // each layer of the network has its own weights, biases, values, and gradients
    int *activationFunctions;   // each layer has its own activation function
    int layerCount;             // number of layers in the network
};
typedef struct Network Network;

// Redeclare network functions
// ACTIVATION CODES: 0 = sigmoid, 1 = relu, 2 = tanh, 3 = linear, 4 = softmax (only for output activation)
Network *createNetwork(int inputNodes, int hiddenLayers, int *hiddenNodes, int outputNodes, int *activationFunctions);
void freeNetwork(Network *network);
void feedForward(Network *network, float *input);
void printNetwork(Network *network);
void backPropagate(Network *network, float *expectedOutputs, float learningRate);
void reinforceNetwork(Network *network, float *expectedOutputs, float learningRate);
void copyNetwork(Network *destination, Network *source);
void exportNetworkJSON(Network *network, char *filename);
Network *importNetworkJSON(char *filename);

#endif
