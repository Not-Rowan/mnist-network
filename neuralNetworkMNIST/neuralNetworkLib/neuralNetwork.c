#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

#include "neuralNetwork.h"

// By: Rowan Rothe

// redefine local functions
void computeGradients(Network *network, float *expectedOutputs);
void gradientDescent(Network *network, float learningRate);
void gradientAscent(Network *network, float learningRate);

// Define network functions

// Activation function (sigmoid)
// input is a vector of values
// the result is stored in a vector passed in as a parameter
void sigmoid(float *values, float *returnValues, int length) {
    for (int i = 0; i < length; i++) {
        returnValues[i] = 1 / (1 + expf(-values[i]));
    }
}

// Derivative of the sigmoid function (uses calculus chain rule)
void sigmoidDerivative(float *values, float *returnValues, int length) {
    float currentSigmoidValue;
    for (int i = 0; i < length; i++) {
        currentSigmoidValue = 1 / (1 + expf(-values[i]));
        returnValues[i] = currentSigmoidValue * (1 - currentSigmoidValue);
    }
}

// Activation function (ReLU)
void relu(float *values, float *returnValues, int length) {
    for (int i = 0; i < length; i++) {
        returnValues[i] = values[i] > 0 ? values[i] : 0;
    }
}

// Derivative of the ReLU function
void reluDerivative(float *values, float *returnValues, int length) {
    for (int i = 0; i < length; i++) {
        returnValues[i] = values[i] > 0 ? 1 : 0;
    }
}

// Activation function (Tanh)
void tanhActivation(float *values, float *returnValues, int length) {
    for (int i = 0; i < length; i++) {
        // prevent overflow by cutting off values at 20
        if (values[i] > 20) {
            returnValues[i] = 1;
        } else if (values[i] < -20) {
            returnValues[i] = -1;
        } else {
            returnValues[i] = (expf(values[i]) - expf(-values[i])) / (expf(values[i]) + expf(-values[i]));
        }
    }
}

// Derivative of the tanh function
void tanhDerivative(float *values, float *returnValues, int length) {
    float currentTanhValue;
    for (int i = 0; i < length; i++) {
        currentTanhValue = (expf(values[i]) - expf(-values[i])) / (expf(values[i]) + expf(-values[i]));
        returnValues[i] = 1 - (currentTanhValue * currentTanhValue);
    }
}

// Activation function (Linear)
void linear(float *values, float *returnValues, int length) {
    for (int i = 0; i < length; i++) {
        returnValues[i] = values[i];
    }
}

// Derivative of the linear function
void linearDerivative(float *values, float *returnValues, int length) {
    // silence unused parameter warning
    (void)values;
    for (int i = 0; i < length; i++) {
        returnValues[i] = 1;
    }
}

// Activation function (Softmax)
// this must take a vecor as input because it is a probability distribution across a vector
// it will also return a vector
void softmax(float *values, float *returnValues, int length) {
    // find max value in the vector
    float max = values[0];
    for (int i = 1; i < length; i++) {
        if (values[i] > max) {
            max = values[i];
        }
    }

    // calculate the softmax function
    float sumExp = 0.0f;
    for (int i = 0; i < length; i++) {
        returnValues[i] = expf(values[i] - max); // subtract max value for numerical stability
        sumExp += returnValues[i];
    }
    for (int i = 0; i < length; i++) {
        returnValues[i] /= sumExp;
    }
}

// Derivative of the softmax function (unneeded because softmax's derivative is the difference between the predicted and expected values)

// Cost function (squared error)
// just like activation functions, this must take a vector as input and return a vector
void squaredError(float *expected, float *actual, float *returnValues, int length) {
    for (int i = 0; i < length; i++) {
        returnValues[i] = (expected[i] - actual[i]) * (expected[i] - actual[i]);
    }
}

// Derivative of the squared error cost function
void squaredErrorDerivative(float *expected, float *actual, float *returnValues, int length) {
    for (int i = 0; i < length; i++) {
        returnValues[i] = 2 * (actual[i] - expected[i]);
    }
}

// Create network
// ACTIVATION CODES: 0 = sigmoid, 1 = relu, 2 = tanh, 3 = linear, 4 = softmax (only for output activation)
// activation functions must be an array of length hiddenLayers + 1. The input layer does not have an activation function
Network *createNetwork(int inputNodes, int hiddenLayers, int *hiddenNodes, int outputNodes, int *activationFunctions) {
    // Create network
    Network *network = malloc(sizeof(Network));
    if (network == NULL) {
        return NULL;
    }
    network->layerCount = hiddenLayers + 2;
    network->layers = malloc(network->layerCount * sizeof(Layer));
    if (network->layers == NULL) {
        freeNetwork(network);
        return NULL;
    }

    // Specify the activation functions for each layer
    // exclude input layer and exclude softmax from hidden layers
    network->activationFunctions = malloc((network->layerCount - 1) * sizeof(int));
    if (network->activationFunctions == NULL) {
        freeNetwork(network);
        return NULL;
    }
    for (int i = 0; i < network->layerCount - 1; i++) {
        if (i != network->layerCount - 2 && activationFunctions[i] == 4) {
            // softmax is only for the output layer
            freeNetwork(network);
            return NULL;
        }
        
        network->activationFunctions[i] = activationFunctions[i];
    }

    // Create input layer
    network->layers[0].neuronCount = inputNodes;
    
    // No incoming weights for the input layer
    network->layers[0].incomingWeights = NULL;
    
    // Initialize input layer (allocate memory) and set bias to random number between -0.3 and 0.3. Input layer value and cost will be set when input is fed forward
    // Biases
    network->layers[0].biases = malloc(network->layers[0].neuronCount * sizeof(float));
    if (network->layers[0].biases == NULL) {
        freeNetwork(network);
        return NULL;
    }
    
    // Values
    network->layers[0].values = malloc(network->layers[0].neuronCount * sizeof(float));
    if (network->layers[0].values == NULL) {
        freeNetwork(network);
        return NULL;
    }
    
    // Gradients
    network->layers[0].gradients = malloc(network->layers[0].neuronCount * sizeof(float));
    if (network->layers[0].gradients == NULL) {
        freeNetwork(network);
        return NULL;
    }
    
    // Set initial values
    for (int inputIndex = 0; inputIndex < inputNodes; inputIndex++) {
        network->layers[0].biases[inputIndex] = ((float)rand() / RAND_MAX) * 0.6 - 0.3;
        network->layers[0].values[inputIndex] = 0;
        network->layers[0].gradients[inputIndex] = 0;
    }


    // Create hidden layers
    for (int currentHiddenLayer = 1; currentHiddenLayer < hiddenLayers + 1; currentHiddenLayer++) {
        network->layers[currentHiddenLayer].neuronCount = hiddenNodes[currentHiddenLayer - 1];
        
        // allocate memory for biases values and gradients
        // Biases
        network->layers[currentHiddenLayer].biases = malloc(network->layers[currentHiddenLayer].neuronCount * sizeof(float));
        if (network->layers[currentHiddenLayer].biases == NULL) {
            freeNetwork(network);
            return NULL;
        }
        
        // Values
        network->layers[currentHiddenLayer].values = malloc(network->layers[currentHiddenLayer].neuronCount * sizeof(float));
        if (network->layers[currentHiddenLayer].values == NULL) {
            freeNetwork(network);
            return NULL;
        }
        
        // Gradients
        network->layers[currentHiddenLayer].gradients = malloc(network->layers[currentHiddenLayer].neuronCount * sizeof(float));
        if (network->layers[currentHiddenLayer].gradients == NULL) {
            freeNetwork(network);
            return NULL;
        }
        
        // allocate the necessary memory for each neuron to have it's own incoming weights array
        network->layers[currentHiddenLayer].incomingWeights = malloc(network->layers[currentHiddenLayer].neuronCount * sizeof(float *));
        if (network->layers[currentHiddenLayer].incomingWeights == NULL) {
            freeNetwork(network);
            return NULL;
        }

        // Initialize hidden layer with random weights and biases between -0.3 and 0.3
        for (int currentHiddenNode = 0; currentHiddenNode < network->layers[currentHiddenLayer].neuronCount; currentHiddenNode++) {
            // allocate memory for these incoming weight arrays
            network->layers[currentHiddenLayer].incomingWeights[currentHiddenNode] = malloc(network->layers[currentHiddenLayer - 1].neuronCount * sizeof(float));
            if (network->layers[currentHiddenLayer].incomingWeights[currentHiddenNode] == NULL) {
                freeNetwork(network);
                return NULL;
            }
            
            // assign random weights to each incoming weight
            for (int previousNeuronIndex = 0; previousNeuronIndex < network->layers[currentHiddenLayer - 1].neuronCount; previousNeuronIndex++) {
                network->layers[currentHiddenLayer].incomingWeights[currentHiddenNode][previousNeuronIndex] = ((float)rand() / RAND_MAX) * 0.6 - 0.3; // Random weights between -0.3 and 0.3
            }
            
            // assign a random bias to the current neuron and set everything else to 0
            network->layers[currentHiddenLayer].biases[currentHiddenNode] = ((float)rand() / RAND_MAX) * 0.6 - 0.3;
            network->layers[currentHiddenLayer].values[currentHiddenNode] = 0;
            network->layers[currentHiddenLayer].gradients[currentHiddenNode] = 0;
        }
    }
    

    // Create output layer
    int outputLayerIndex = network->layerCount - 1;
    network->layers[outputLayerIndex].neuronCount = outputNodes;
    
    // allocate memory for biases, values, and gradients
    network->layers[outputLayerIndex].biases = malloc(network->layers[outputLayerIndex].neuronCount * sizeof(float));
    if (network->layers[outputLayerIndex].biases == NULL) {
        freeNetwork(network);
        return NULL;
    }
    
    network->layers[outputLayerIndex].values = malloc(network->layers[outputLayerIndex].neuronCount * sizeof(float));
    if (network->layers[outputLayerIndex].values == NULL) {
        freeNetwork(network);
        return NULL;
    }
    
    network->layers[outputLayerIndex].gradients = malloc(network->layers[outputLayerIndex].neuronCount * sizeof(float));
    if (network->layers[outputLayerIndex].gradients == NULL) {
        freeNetwork(network);
        return NULL;
    }
    
    // allocate memory so each neuron can have its own incoming weights array
    network->layers[outputLayerIndex].incomingWeights = malloc(network->layers[outputLayerIndex].neuronCount * sizeof(float *));
    if (network->layers[outputLayerIndex].incomingWeights == NULL) {
        freeNetwork(network);
        return NULL;
    }
    
    // initialize output layer with random weights and biases between -0.3 and 0.3
    for (int currentOutputNode = 0; currentOutputNode < outputNodes; currentOutputNode++) {
        // allocate memory for those incoming weights arrays
        network->layers[outputLayerIndex].incomingWeights[currentOutputNode] = malloc(network->layers[outputLayerIndex - 1].neuronCount * sizeof(float));
        if (network->layers[outputLayerIndex].incomingWeights[currentOutputNode] == NULL) {
            freeNetwork(network);
            return NULL;
        }
        
        for (int previousNodeIndex = 0; previousNodeIndex < network->layers[outputLayerIndex - 1].neuronCount; previousNodeIndex++) {
            // set the weights to a random value between -0.3 and 0.3
            network->layers[outputLayerIndex].incomingWeights[currentOutputNode][previousNodeIndex] = ((float)rand() / RAND_MAX) * 0.6 - 0.3;
        }
        
        // set the bias for the output node to random value between -0.3 and 0.3 and everything else to 0
        network->layers[outputLayerIndex].biases[currentOutputNode] = ((float)rand() / RAND_MAX) * 0.6 - 0.3;
        network->layers[outputLayerIndex].values[currentOutputNode] = 0;
        network->layers[outputLayerIndex].gradients[currentOutputNode] = 0;
    }

    return network;
}

// free network
void freeNetwork(Network *network) {
    for (int currentLayer = 0; currentLayer < network->layerCount; currentLayer++) {
        if (network->layers[currentLayer].values != NULL) {
            free(network->layers[currentLayer].values);
            network->layers[currentLayer].values = NULL;
        }
        
        if (network->layers[currentLayer].biases != NULL) {
            free(network->layers[currentLayer].biases);
            network->layers[currentLayer].biases = NULL;
        }
        
        if (network->layers[currentLayer].gradients != NULL) {
            free(network->layers[currentLayer].gradients);
            network->layers[currentLayer].gradients = NULL;
        }
        
        if (network->layers[currentLayer].incomingWeights != NULL) {
            for (int currentNode = 0; currentNode < network->layers[currentLayer].neuronCount; currentNode++) {
                if (network->layers[currentLayer].incomingWeights[currentNode] != NULL) {
                    free(network->layers[currentLayer].incomingWeights[currentNode]);
                    network->layers[currentLayer].incomingWeights[currentNode] = NULL;
                }
            }
            
            free(network->layers[currentLayer].incomingWeights);
            network->layers[currentLayer].incomingWeights = NULL;
        }
    }
    if (network->activationFunctions != NULL) {
        free(network->activationFunctions);
    }
    if (network->layers != NULL) {
        free(network->layers);
    }
    if (network != NULL) {
        free(network);
    }
}

// feed forward by multiplying incoming value by incoming weights and adding the current neuron's bias
void feedForward(Network *network, float *input) {
    // Set input layer values to input
    for (int inputNeuron = 0; inputNeuron < network->layers[0].neuronCount; inputNeuron++) {
        network->layers[0].values[inputNeuron] = input[inputNeuron];
    }

    // Calculate and set hidden layer and output layer values using the previous neuron's value multiplied by the neuron's incoming weights and add the current neuron's bias
    // Pass this into an activation function to get the final value
    for (int currentLayer = 1; currentLayer < network->layerCount; currentLayer++) {
        // init weighted sum array
        float weightedSumArray[network->layers[currentLayer].neuronCount];

        // first calculate the weighted sum for each neuron in the current layer (weights plus bias plus previous layer value)
        for (int currentNeuron = 0; currentNeuron < network->layers[currentLayer].neuronCount; currentNeuron++) {
            float valueWeightSum = 0;
            for (int previousNeuronIndex = 0; previousNeuronIndex < network->layers[currentLayer - 1].neuronCount; previousNeuronIndex++) {
                valueWeightSum += network->layers[currentLayer - 1].values[previousNeuronIndex] * network->layers[currentLayer].incomingWeights[currentNeuron][previousNeuronIndex];
            }
            weightedSumArray[currentNeuron] = valueWeightSum + network->layers[currentLayer].biases[currentNeuron];
        }

        // then apply the activation function to the weighted sum
        // the activation function is specified in the activationFunctions array
        switch (network->activationFunctions[currentLayer-1]) {
            case 0:
                // sigmoid
                sigmoid(weightedSumArray, network->layers[currentLayer].values, network->layers[currentLayer].neuronCount);
                break;
            case 1:
                // relu
                relu(weightedSumArray, network->layers[currentLayer].values, network->layers[currentLayer].neuronCount);
                break;
            case 2:
                // tanh
                tanhActivation(weightedSumArray, network->layers[currentLayer].values, network->layers[currentLayer].neuronCount);
                break;
            case 3:
                // linear
                linear(weightedSumArray, network->layers[currentLayer].values, network->layers[currentLayer].neuronCount);
                break;
            case 4:
                // softmax
                softmax(weightedSumArray, network->layers[currentLayer].values, network->layers[currentLayer].neuronCount);
                break;
            default:
                // default to sigmoid
                sigmoid(weightedSumArray, network->layers[currentLayer].values, network->layers[currentLayer].neuronCount);
                break;
        }
    }
}

// print the structure of the network including layer num, node num, node values, node biases, and incoming weights
void printNetwork(Network *network) {
    printf("---Network---\n");
    for (int currentLayer = 0; currentLayer < network->layerCount; currentLayer++) {
        printf("Layer %d", currentLayer);
        if (currentLayer == 0) {
            printf(" (input):\n");
        } else if (currentLayer == network->layerCount - 1) {
            printf(" (output):\n");
        } else {
            printf(" (hidden):\n");
        }

        for (int currentNeuron = 0; currentNeuron < network->layers[currentLayer].neuronCount; currentNeuron++) {
            printf("    Node %d:\n        Value: %f\n        Bias: %f\n", currentNeuron, network->layers[currentLayer].values[currentNeuron], network->layers[currentLayer].biases[currentNeuron]);
            // skip printing weights if input layer because the input layer has no incoming weights
            if (currentLayer == 0) continue;
            printf("        Incoming Weights: {");
            for (int previousNeuronIndex = 0; previousNeuronIndex < network->layers[currentLayer - 1].neuronCount; previousNeuronIndex++) {
                printf("%f", network->layers[currentLayer].incomingWeights[currentNeuron][previousNeuronIndex]);
                if (previousNeuronIndex == network->layers[currentLayer - 1].neuronCount - 1) continue;
                printf(", ");
            }
            printf("}\n");
        }
    }
}

// Backpropagate using gradient descent.
// compute the gradients for each parameter in the network and apply the error to the network weights using gradient descent
void backPropagate(Network *network, float *expectedOutputs, float learningRate) {
    // compute the gradients for each layer
    computeGradients(network, expectedOutputs);

    // apply error to network weights using gradient descent
    gradientDescent(network, learningRate);
}

// Function to reinforce the network using gradient ascent
// This is the opposite of gradient descent
void reinforceNetwork(Network *network, float *expectedOutputs, float learningRate) {
    // propogate the gradient backwards through each layer
    computeGradients(network, expectedOutputs);

    // apply error to network weights using gradient descent
    gradientAscent(network, learningRate);
}

// Function to compute the gradients for each parameter in the network
void computeGradients(Network *network, float *expectedOutputs) {
    // calculate the gradients for the output layer
    int outputLayerIndex = network->layerCount - 1;
    if (network->activationFunctions[outputLayerIndex-1] == 4) {
        // softmax (softmax will automatically use cross entropy as the cost function)

        // cross entropy gradient is calculated as the difference between the target value and the predicted value for that output
        // predicted_value - target_value
        for (int outputNeuron = 0; outputNeuron < network->layers[outputLayerIndex].neuronCount; outputNeuron++) {
            network->layers[outputLayerIndex].gradients[outputNeuron] = network->layers[outputLayerIndex].values[outputNeuron] - expectedOutputs[outputNeuron];
        }
    } else {
        // other activation functions (use squared error cost function)

        // first calculate the cost of each neuron in the output layer
        float squaredErrorCosts[network->layers[outputLayerIndex].neuronCount];
        squaredErrorDerivative(expectedOutputs, network->layers[outputLayerIndex].values, squaredErrorCosts, network->layers[outputLayerIndex].neuronCount);

        for (int outputNeuron = 0; outputNeuron < network->layers[outputLayerIndex].neuronCount; outputNeuron++) {
            network->layers[outputLayerIndex].gradients[outputNeuron] = squaredErrorCosts[outputNeuron];
        }

        // then calculate the derivative of the activation function for each neuron in the output layer
        // this is stored in the gradients array for each neuron
        switch (network->activationFunctions[outputLayerIndex-1]) {
            case 0:
                // sigmoid
                sigmoidDerivative(network->layers[outputLayerIndex].values, network->layers[outputLayerIndex].gradients, network->layers[outputLayerIndex].neuronCount);
                break;
            case 1:
                // relu
                reluDerivative(network->layers[outputLayerIndex].values, network->layers[outputLayerIndex].gradients, network->layers[outputLayerIndex].neuronCount);
                break;
            case 2:
                // tanh
                tanhDerivative(network->layers[outputLayerIndex].values, network->layers[outputLayerIndex].gradients, network->layers[outputLayerIndex].neuronCount);
                break;
            case 3:
                // linear
                linearDerivative(network->layers[outputLayerIndex].values, network->layers[outputLayerIndex].gradients, network->layers[outputLayerIndex].neuronCount);
                break;
            default:
                // default to sigmoid
                sigmoidDerivative(network->layers[outputLayerIndex].values, network->layers[outputLayerIndex].gradients, network->layers[outputLayerIndex].neuronCount);
                break;
        }

        // multiply the error by the derivative of the activation function for each neuron in the output layer and apply that to the gradient (this will be propogated backwards / applied later)
        for (int outputNeuron = 0; outputNeuron < network->layers[outputLayerIndex].neuronCount; outputNeuron++) {
            network->layers[outputLayerIndex].gradients[outputNeuron] = network->layers[outputLayerIndex].gradients[outputNeuron] * squaredErrorCosts[outputNeuron];
        }
    }


    // hidden layers
    for (int currentLayer = network->layerCount - 2; currentLayer > 0; currentLayer--) {
        float weightedErrorValues[network->layers[currentLayer].neuronCount];

        // calculate the weighted error for each neuron in the current layer
        for (int currentNeuron = 0; currentNeuron < network->layers[currentLayer].neuronCount; currentNeuron++) {
            float errorWeightSum = 0.0;
            for (int nextNeuronIndex = 0; nextNeuronIndex < network->layers[currentLayer + 1].neuronCount; nextNeuronIndex++) {
                errorWeightSum += network->layers[currentLayer + 1].gradients[nextNeuronIndex] * network->layers[currentLayer + 1].incomingWeights[nextNeuronIndex][currentNeuron];
            }
            weightedErrorValues[currentNeuron] = errorWeightSum;
        }

        // then calculate the derivative of the activation function for each neuron in the current layer
        switch (network->activationFunctions[currentLayer-1]) {
            case 0:
                // sigmoid
                sigmoidDerivative(network->layers[currentLayer].values, network->layers[currentLayer].gradients, network->layers[currentLayer].neuronCount);
                break;
            case 1:
                // relu
                reluDerivative(network->layers[currentLayer].values, network->layers[currentLayer].gradients, network->layers[currentLayer].neuronCount);
                break;
            case 2:
                // tanh
                tanhDerivative(network->layers[currentLayer].values, network->layers[currentLayer].gradients, network->layers[currentLayer].neuronCount);
                break;
            case 3:
                // linear
                linearDerivative(network->layers[currentLayer].values, network->layers[currentLayer].gradients, network->layers[currentLayer].neuronCount);
                break;
            default:
                // default to sigmoid
                sigmoidDerivative(network->layers[currentLayer].values, network->layers[currentLayer].gradients, network->layers[currentLayer].neuronCount);
                break;
        }

        // then multiply the error by the derivative of the activation function for each neuron in the current layer and apply that to the gradient (this will be propogated backwards / applied later)
        for (int currentNeuron = 0; currentNeuron < network->layers[currentLayer].neuronCount; currentNeuron++) {
            network->layers[currentLayer].gradients[currentNeuron] = weightedErrorValues[currentNeuron] * network->layers[currentLayer].gradients[currentNeuron];
        }
    }
}

// Function to descend a gradient (gradient descent)
// equation: theta = theta - learningRate * gradient
// theta is the parameter (bias or weight), learningRate is the learning rate, and gradient is the gradient of the parameter
void gradientDescent(Network *network, float learningRate) {
    for (int currentLayer = network->layerCount - 1; currentLayer > 0; currentLayer--) {
        for (int currentNeuron = 0; currentNeuron < network->layers[currentLayer].neuronCount; currentNeuron++) {
            network->layers[currentLayer].biases[currentNeuron] -= network->layers[currentLayer].gradients[currentNeuron] * learningRate;

            for (int previousNeuronIndex = 0; previousNeuronIndex < network->layers[currentLayer - 1].neuronCount; previousNeuronIndex++) {
                network->layers[currentLayer].incomingWeights[currentNeuron][previousNeuronIndex] -= network->layers[currentLayer].gradients[currentNeuron] * network->layers[currentLayer - 1].values[previousNeuronIndex] * learningRate;
            }
        }
    }

    // clear the gradients after applying them
    for (int currentLayer = 0; currentLayer < network->layerCount; currentLayer++) {
        for (int currentNeuron = 0; currentNeuron < network->layers[currentLayer].neuronCount; currentNeuron++) {
            network->layers[currentLayer].gradients[currentNeuron] = 0;
        }
    }
}

// Function to ascent a gradient (gradient descent)
// The opposite of gradient descent where to calculate the gradient, we move in the opposite direction
// equation: theta = theta + learningRate * gradient
// theta is the parameter (bias or weight), learningRate is the learning rate, and gradient is the gradient of the parameter
void gradientAscent(Network *network, float learningRate) {
    for (int currentLayer = network->layerCount - 1; currentLayer > 0; currentLayer--) {
        for (int currentNeuron = 0; currentNeuron < network->layers[currentLayer].neuronCount; currentNeuron++) {
            network->layers[currentLayer].biases[currentNeuron] += network->layers[currentLayer].gradients[currentNeuron] * learningRate;

            for (int previousNeuronIndex = 0; previousNeuronIndex < network->layers[currentLayer - 1].neuronCount; previousNeuronIndex++) {
                network->layers[currentLayer].incomingWeights[currentNeuron][previousNeuronIndex] += network->layers[currentLayer].gradients[currentNeuron] * network->layers[currentLayer - 1].values[previousNeuronIndex] * learningRate;
            }
        }
    }

    // clear the gradients after applying them
    for (int currentLayer = 0; currentLayer < network->layerCount; currentLayer++) {
        for (int currentNeuron = 0; currentNeuron < network->layers[currentLayer].neuronCount; currentNeuron++) {
            network->layers[currentLayer].gradients[currentNeuron] = 0;
        }
    }
}

// Function to copy one network to another as long as the structure is the same
void copyNetwork(Network *destination, Network *source) {
    // check if the layer counts are the same
    if (destination->layerCount != source->layerCount) {
        return;
    }

    // copy the biases, values, gradients, and incoming weights
    for (int currentLayer = 0; currentLayer < source->layerCount; currentLayer++) {
        // check if the neuron counts are the same
        if (destination->layers[currentLayer].neuronCount != source->layers[currentLayer].neuronCount) {
            return;
        }

        // copy the biases and incoming weights
        for (int currentNeuron = 0; currentNeuron < source->layers[currentLayer].neuronCount; currentNeuron++) {
            destination->layers[currentLayer].biases[currentNeuron] = source->layers[currentLayer].biases[currentNeuron];
            if (currentLayer != 0) {
                for (int previousNeuronIndex = 0; previousNeuronIndex < source->layers[currentLayer - 1].neuronCount; previousNeuronIndex++) {
                    destination->layers[currentLayer].incomingWeights[currentNeuron][previousNeuronIndex] = source->layers[currentLayer].incomingWeights[currentNeuron][previousNeuronIndex];
                }
            }
        }
    }
}

// Export network for loading later (.json file format)
void exportNetworkJSON(Network *network, char *filename) {
    // open file
    FILE *fp = fopen(filename, "w");
    if (fp == NULL) {
        return;
    }

    // write to file
    fprintf(fp, "{\n");
    fprintf(fp, "    \"layerCount\": %d,\n", network->layerCount);
    fprintf(fp, "    \"activationFunctions\": [\n");
    for (int i = 0; i < network->layerCount - 1; i++) {
        fprintf(fp, "        %d", network->activationFunctions[i]);
        if (i == network->layerCount - 2) {
            fprintf(fp, "\n");
        } else {
            fprintf(fp, ",\n");
        }
    }
    fprintf(fp, "    ],\n");
    fprintf(fp, "    \"layers\": [\n");
    for (int i = 0; i < network->layerCount; i++) {
        fprintf(fp, "        {\n");
        fprintf(fp, "            \"neuronCount\": %d,\n", network->layers[i].neuronCount);
        fprintf(fp, "            \"neurons\": [\n");
        for (int j = 0; j < network->layers[i].neuronCount; j++) {
            fprintf(fp, "                {\n");
            if (i != 0) {
                fprintf(fp, "                    \"incomingWeights\": [");
                for (int k = 0; k < network->layers[i - 1].neuronCount; k++) {
                    fprintf(fp, "%f", network->layers[i].incomingWeights[j][k]);
                    if (k == network->layers[i - 1].neuronCount - 1) {
                        fprintf(fp, "],\n");
                    } else {
                        fprintf(fp, ", ");
                    }
                }
            }

            fprintf(fp, "                    \"bias\": %f,\n", network->layers[i].biases[j]);
            fprintf(fp, "                    \"value\": %f,\n", network->layers[i].values[j]);
            fprintf(fp, "                    \"gradients\": %f\n", network->layers[i].gradients[j]);
            if (j == network->layers[i].neuronCount - 1) {
                fprintf(fp, "                }\n");
            } else {
                fprintf(fp, "                },\n");
            }
        }
        if (i == network->layerCount - 1) {
            fprintf(fp, "            ]\n");
        } else {
            fprintf(fp, "            ]\n");
        }
        if (i == network->layerCount - 1) {
            fprintf(fp, "        }\n");
        } else {
            fprintf(fp, "        },\n");
        }
    }
    fprintf(fp, "    ]\n");
    fprintf(fp, "}\n");

    // close file
    fclose(fp);
}

// Import network from file (.json file format)
Network *importNetworkJSON(char *filename) {
    // open file
    FILE *fp = fopen(filename, "r");
    if (fp == NULL) {
        return NULL;
    }

    // create the network
    Network *network = (Network *)malloc(sizeof(Network));
    if (network == NULL) {
        return NULL;
    }

    // read from the file and set the network values based on input
    fscanf(fp, "{\n");
    fscanf(fp, "    \"layerCount\": %d,\n", &network->layerCount);
    fscanf(fp, "    \"activationFunctions\": [\n");
    network->activationFunctions = (int *)malloc((network->layerCount - 1) * sizeof(int));
    if (network->activationFunctions == NULL) {
        freeNetwork(network);
        return NULL;
    }
    for (int i = 0; i < network->layerCount - 1; i++) {
        fscanf(fp, "        %d", &network->activationFunctions[i]);
        if (i == network->layerCount - 2) {
            fscanf(fp, "\n");
        } else {
            fscanf(fp, ",\n");
        }
    }
    fscanf(fp, "    ],\n");
    fscanf(fp, "    \"layers\": [\n");
    network->layers = (Layer *)malloc(network->layerCount * sizeof(Layer));
    if (network->layers == NULL) {
        freeNetwork(network);
        return NULL;
    }
    for (int i = 0; i < network->layerCount; i++) {
        fscanf(fp, "        {\n");
        fscanf(fp, "            \"neuronCount\": %d,\n", &network->layers[i].neuronCount);
        fscanf(fp, "            \"neurons\": [\n");
        
        // allocate memory for biases, values, and gradients
        network->layers[i].biases = malloc(network->layers[i].neuronCount * sizeof(float));
        if (network->layers[i].biases == NULL) {
            freeNetwork(network);
            return NULL;
        }
        
        network->layers[i].values = malloc(network->layers[i].neuronCount * sizeof(float));
        if (network->layers[i].values == NULL) {
            freeNetwork(network);
            return NULL;
        }
        
        network->layers[i].gradients = malloc(network->layers[i].neuronCount * sizeof(float));
        if (network->layers[i].gradients == NULL) {
            freeNetwork(network);
            return NULL;
        }
        
        // allocate memory for incoming weights arrays for each neuron
        network->layers[i].incomingWeights = malloc(network->layers[i].neuronCount * sizeof(float *));
        if (network->layers[i].incomingWeights == NULL) {
            freeNetwork(network);
            return NULL;
        }
        
        // make incoming weights null if first layer
        if (i == 0) {
            network->layers[i].incomingWeights = NULL;
        }
        
        for (int j = 0; j < network->layers[i].neuronCount; j++) {
            fscanf(fp, "                {\n");
            if (i != 0) {
                fscanf(fp, "                    \"incomingWeights\": [");
                network->layers[i].incomingWeights[j] = malloc(network->layers[i - 1].neuronCount * sizeof(float));
                if (network->layers[i].incomingWeights[j] == NULL) {
                    freeNetwork(network);
                    return NULL;
                }
                for (int k = 0; k < network->layers[i - 1].neuronCount; k++) {
                    fscanf(fp, "%f", &network->layers[i].incomingWeights[j][k]);
                    if (k == network->layers[i - 1].neuronCount - 1) {
                        fscanf(fp, "],\n");
                    } else {
                        fscanf(fp, ", ");
                    }
                }
            }
            
            fscanf(fp, "                    \"bias\": %f,\n", &network->layers[i].biases[j]);
            fscanf(fp, "                    \"value\": %f,\n", &network->layers[i].values[j]);
            fscanf(fp, "                    \"gradients\": %f\n", &network->layers[i].gradients[j]);
            if (j == network->layers[i].neuronCount - 1) {
                fscanf(fp, "                }\n");
            } else {
                fscanf(fp, "                },\n");
            }
        }
        if (i == network->layerCount - 1) {
            fscanf(fp, "            ]\n");
        } else {
            fscanf(fp, "            ]\n");
        }
        if (i == network->layerCount - 1) {
            fscanf(fp, "        }\n");
        } else {
            fscanf(fp, "        },\n");
        }
    }
    fscanf(fp, "    ]\n");
    fscanf(fp, "}\n");

    // close the file
    fclose(fp);

    return network;
}

