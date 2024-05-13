#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <time.h>
#include <math.h>
#include <pthread.h>

#include "neuralNetworkLib/neuralNetwork.h"

// Best: 92% training accuracy and 95% test accuracy (epoch 10, learning rate 0.03, hidden nodes 500, 1 layers). 2292 seconds
// Training again with 80 epochs for refinement. ended up with 96% training accuracy and 96% test accuracy
// Training again with 80 epochs for refinement. ended up with 96% training accuracy and 97% test accuracy

// Softmax with cross entropy loss function. 92% training accuracy and 93% test accuracy (epoch 100, learning rate 0.03, hidden nodes 500, 1 layers)

// declare network constants
#define INPUT_NODES 784  // each image is 28x28 pixels, so there are 784 pixels in total
#define HIDDEN_LAYERS 1 // x hidden layers for this network
#define HIDDEN_NODES (int[]){500} // x hidden layers with x nodes in each layer
#define OUTPUT_NODES 10 // since we are using the mnist data set, the output will be a number between 0 and 9
#define LEARNING_RATE 0.03f //0.03f // learning rate for gradient descent

#define HIDDEN_ACTIVATION 1 // 0 for sigmoid, 1 for relu, 2 for tanh (3 for softmax isnt supported for hidden layers)
#define OUTPUT_ACTIVATION 0 // 0 for sigmoid, 1 for relu, 2 for tanh 3 for linear, 4 for softmax

#define TRAINING_DATA_SETS 60000 // 60,000 mnist training data sets
#define MINI_BATCH_SIZE 100 // each mini batch is 100 data sets
#define TESTING_DATA_SETS 10000 // 10,000 mnist test data sets
#define EPOCH 100 //10 // go through the training data set x times


// debug constants (0 for false and 1 for true)
#define LOG_ACCURACY 0

// Redeclare network functions
void testNetwork(float *inputData, int inputDataSize, float *outputData, int outputDataSize, int dataLength);
void shuffleData(float *inputData, int inputDataSize, float *outputData, int outputDataSize, int dataLength);
void addNoise(float *inputData, int dataLength, int noiseLevel);
void translateData(float *inputData, int dataWidth, int dataHeight, int offsetX, int offsetY);
void scaleData(float *inputData, int dataWidth, int dataHeight, float scale);
void rotateData(float *inputData, int dataWidth, int dataHeight, float angle);
void displayImage(float *inputData, int dataWidth, int dataHeight);

// define threading variables, structs, and functions
struct ThreadArgs {
    int epoch;
    int batchNum;
    int batchSize;
    float *batchDataInputPtr;
    float *batchDataOutputPtr;
};
typedef struct ThreadArgs ThreadArgs;
void *trainOnBatch(void *arg);
pthread_mutex_t lock;
Network *network;
pthread_t threads[TRAINING_DATA_SETS / MINI_BATCH_SIZE];
int threadsCompleted = 0;
int totalThreadsCompleted = 0;
float totalTrainingAccuracy = 0;

// other global variables
int testCorrect = 0;

int main(void) {
    // declare file pointer
    FILE *fp;

    // seed random number generator
    printf("Seeding random number generator\n");
    srand(time(NULL));

    // clear accuracy.csv file
    if (LOG_ACCURACY) {
        printf("Clearing accuracy.csv file\n");
        fp = fopen("accuracy.csv", "w");
        if (fp == NULL) {
            perror("Could not open accuracy.csv");
            return 1;
        }
        fclose(fp);
    }

    // ask user if they would like to load a network
    char load;
    printf("Would you like to load a network? (y/n): ");
    scanf("%c", &load);
    if (load == 'y') {
        printf("Loading network...\n");
        network = importNetworkJSON("network.json");
        if (network == NULL) {
            perror("Error loading network");
            freeNetwork(network);
            return 1;
        }

        printf("Network loaded\n");

        // consume the newline character
        while ((load = getchar()) != '\n');

        printf("Would you like to continue training the network (otherwise the program will begin taking user input)? (y/n): ");
        char continueTraining;
        scanf("%c", &continueTraining);
        if (continueTraining == 'y') {
            // do nothing and pass through to training
            printf("Continuing training...\n");
        } else if (continueTraining == 'n') {
            // test network with user input
            goto testWithUserInput;
        } else {
            printf("Invalid input\n");
            return 1;
        }
    } else if (load != 'n') {
        printf("Invalid input\n");
        return 1;
    }


    // training data (mnist_train.csv)
    // in the training sets for mnist, the first column is the label of the image and the rest of the columns are the pixel values
    // each image is 28x28 pixels, so there are 784 pixels in total
    // the label is the number that the image represents
    // the pixel values are between 0 and 255 (but we will normalize them to be between 0 and 1 by dividing by 255)
    printf("Opening mnist_train.csv file\n");
    fp = fopen("mnistData/mnist_train.csv", "r");
    if (fp == NULL) {
        perror("Error opening file");
        return 1;
    }

    // read training data (csv format)
    printf("Reading training data\n");
    float (*trainingInput)[INPUT_NODES] = malloc(TRAINING_DATA_SETS * sizeof(*trainingInput));
    float (*trainingOutput)[OUTPUT_NODES] = malloc(TRAINING_DATA_SETS * sizeof(*trainingOutput));
    if (trainingInput == NULL || trainingOutput == NULL) {
        perror("Error allocating memory");
        return 1;
    }
    
    // go through all the rows (except the first row which is the header row)
    // skip the first row
    char c;
    while ((c = fgetc(fp)) != '\n') {
        continue;
    }

    // go through the rest of the rows
    for (int rows = 0; rows < TRAINING_DATA_SETS; rows++) {
        // read the label values
        int label;
        fscanf(fp, "%d,", &label);

        // set the label value to 1 and the rest to 0 (one hot encoding)
        for (int cols = 0; cols < OUTPUT_NODES; cols++) {
            if (cols == label) {
                trainingOutput[rows][cols] = 1;
            } else {
                trainingOutput[rows][cols] = 0;
            }
        }
        
        // read the pixel values
        for (int cols = 0; cols < INPUT_NODES; cols++) {
            if (cols == INPUT_NODES - 1) {
                fscanf(fp, "%f\n", &trainingInput[rows][cols]);
            } else {
                fscanf(fp, "%f,", &trainingInput[rows][cols]);
            }
        }

        // normalize the pixel values to be between 0 and 1
        for (int cols = 0; cols < INPUT_NODES; cols++) {
            trainingInput[rows][cols] /= 255;
        }
    }

    // close mnist_train.csv file
    fclose(fp);

    // open mnist_test.csv file for testing
    printf("Opening mnist_test.csv file\n");
    fp = fopen("mnistData/mnist_test.csv", "r");
    if (fp == NULL) {
        perror("Error opening file");
        return 1;
    }

    // read test data and skip the first row
    printf("Reading test data\n");
    float (*testInput)[INPUT_NODES] = malloc(TESTING_DATA_SETS * sizeof(*testInput));
    float (*testOutput)[OUTPUT_NODES] = malloc(TESTING_DATA_SETS * sizeof(*testOutput));
    if (testInput == NULL || testOutput == NULL) {
        perror("Error allocating memory");
        return 1;
    }

    // go through all the rows (except the first row which is the header row)
    // skip the first row
    while ((c = fgetc(fp)) != '\n') {
        continue;
    }

    // go through the rest of the rows
    for (int rows = 0; rows < TESTING_DATA_SETS; rows++) {
        // read the label values
        int label;
        fscanf(fp, "%d,", &label);

        // set the label value to 1 and the rest to 0 (one hot encoding)
        for (int cols = 0; cols < OUTPUT_NODES; cols++) {
            if (cols == label) {
                testOutput[rows][cols] = 1;
            } else {
                testOutput[rows][cols] = 0;
            }
        }
        
        // read the pixel values
        for (int cols = 0; cols < INPUT_NODES; cols++) {
            if (cols == INPUT_NODES - 1) {
                fscanf(fp, "%f\n", &testInput[rows][cols]);
            } else {
                fscanf(fp, "%f,", &testInput[rows][cols]);
            }
        }

        // normalize the pixel values to be between 0 and 1
        for (int cols = 0; cols < INPUT_NODES; cols++) {
            testInput[rows][cols] /= 255;
        }
    }

    // close mnist_test.csv file
    fclose(fp);


    // initialize thread lock
    printf("Initializing thread parameters\n");

    if (pthread_mutex_init(&lock, NULL) != 0) {
        perror("failed to init mutex lock");
        return 1;
    }

    // we now have read in the training data and the test data into the program
    // we will now create a network and train it using the training data
    // then we will test the network using the test data

    // start timing the process
    clock_t start, end;
    double cpu_time_used;
    start = clock();

    // create network
    if (load == 'n') {
        printf("Creating network\n");
        network = createNetwork(INPUT_NODES, HIDDEN_LAYERS, HIDDEN_NODES, OUTPUT_NODES);
        if (network == NULL) {
            perror("Error creating network");
            return 1;
        }
    }

    // start training the network
    printf("Training network...\n\n");

    // create new variables to hold processed training data
    float (*processedTrainingInput)[INPUT_NODES] = malloc(TRAINING_DATA_SETS * sizeof(*processedTrainingInput));
    float (*processedTrainingOutput)[OUTPUT_NODES] = malloc(TRAINING_DATA_SETS * sizeof(*processedTrainingOutput));
    if (processedTrainingInput == NULL || processedTrainingOutput == NULL) {
        perror("Error allocating memory");
        return 1;
    }

    // create new variables to hold processed test data
    float (*processedTestInput)[INPUT_NODES] = malloc(TESTING_DATA_SETS * sizeof(*processedTestInput));
    float (*processedTestOutput)[OUTPUT_NODES] = malloc(TESTING_DATA_SETS * sizeof(*processedTestOutput));
    if (processedTestInput == NULL || processedTestOutput == NULL) {
        perror("Error allocating memory");
        return 1;
    }

    // copy test data to processed test data
    for (int i = 0; i < TESTING_DATA_SETS; i++) {
        for (int j = 0; j < INPUT_NODES; j++) {
            processedTestInput[i][j] = testInput[i][j];
        }
        for (int j = 0; j < OUTPUT_NODES; j++) {
            processedTestOutput[i][j] = testOutput[i][j];
        }
    }

    // for each epoch, run the mini batch threads to train the network faster
    int threadCount = TRAINING_DATA_SETS / MINI_BATCH_SIZE;
    for (int epoch = 0; epoch < EPOCH; epoch++) {
        // pre-process training data
        printf("Pre-Processing training data for epoch %d\n", epoch);

        // copy training data to processed training data
        for (int currentDataSet = 0; currentDataSet < TRAINING_DATA_SETS; currentDataSet++) {
            for (int currentInputNode = 0; currentInputNode < INPUT_NODES; currentInputNode++) {
                processedTrainingInput[currentDataSet][currentInputNode] = trainingInput[currentDataSet][currentInputNode];
            }
            for (int currentOutputNode = 0; currentOutputNode < OUTPUT_NODES; currentOutputNode++) {
                processedTrainingOutput[currentDataSet][currentOutputNode] = trainingOutput[currentDataSet][currentOutputNode];
            }
        }

        // shuffle data and apply edits to the images (scaling, rotating, translating, and adding noise)
        shuffleData(&processedTrainingInput[0][0], INPUT_NODES, &processedTrainingOutput[0][0], OUTPUT_NODES, TRAINING_DATA_SETS);
        for (int currentDataSet = 0; currentDataSet < TRAINING_DATA_SETS; currentDataSet++) {
            // randomly scale the image between 0.8 and 1.2
            float scale = (float)rand() / RAND_MAX * 0.4 + 0.8;
            scaleData(&processedTrainingInput[currentDataSet][0], (int)sqrt(INPUT_NODES), (int)sqrt(INPUT_NODES), scale);

            // randomly rotate the image between -15 and 15 degrees
            float angle = (float)rand() / RAND_MAX * 30 - 15;
            rotateData(&processedTrainingInput[currentDataSet][0], (int)sqrt(INPUT_NODES), (int)sqrt(INPUT_NODES), angle);
            
            // randomly translate the image between -2 and 2 pixels
            int offsetX = rand() % 4 - 2;
            int offsetY = rand() % 4 - 2;
            translateData(&processedTrainingInput[currentDataSet][0], (int)sqrt(INPUT_NODES), (int)sqrt(INPUT_NODES), offsetX, offsetY);

            // add noise to the current image
            addNoise(&processedTrainingInput[currentDataSet][0], INPUT_NODES, 1);
        }

        // run mini batch threads
        printf("Running mini batch threads...\n");
        for (int batch = 0; batch < threadCount; batch++) {
            // initialize the arguments struct
            ThreadArgs threadArgs;
            threadArgs.epoch = epoch;
            threadArgs.batchNum = batch;
            threadArgs.batchSize = MINI_BATCH_SIZE;
            threadArgs.batchDataInputPtr = &(processedTrainingInput[batch * MINI_BATCH_SIZE][0]);
            threadArgs.batchDataOutputPtr = &(processedTrainingOutput[batch * MINI_BATCH_SIZE][0]);

            if (pthread_create(&(threads[batch]), NULL, &trainOnBatch, &threadArgs) != 0) {
                perror("could not create thread");
                return 1;
            }
        }

        // rejoin the threads
        for (int threadNum = 0; threadNum < threadCount; threadNum++) {
            pthread_join(threads[threadNum], NULL);
        }

        // clear threads completed variable
        threadsCompleted = 0;

    
        // test current network
        printf("\n\n");
        // copy testing data to processed testing data
        for (int currentDataSet = 0; currentDataSet < TESTING_DATA_SETS; currentDataSet++) {
            for (int currentInputNode = 0; currentInputNode < INPUT_NODES; currentInputNode++) {
                processedTestInput[currentDataSet][currentInputNode] = testInput[currentDataSet][currentInputNode];
            }
            for (int currentOutputNode = 0; currentOutputNode < OUTPUT_NODES; currentOutputNode++) {
                processedTestOutput[currentDataSet][currentOutputNode] = testOutput[currentDataSet][currentOutputNode];
            }
        }

        // run data through the network
        testNetwork(&processedTestInput[0][0], INPUT_NODES, &processedTestOutput[0][0], OUTPUT_NODES, TESTING_DATA_SETS);

        // print accuracy in percentage
        printf("Testing accuracy: %f%%\n\n", (float)testCorrect / TESTING_DATA_SETS * 100);
        testCorrect = 0;
    }

    // destroy thread mutex lock
    pthread_mutex_destroy(&lock);

    // free processed training data variable memory
    free(processedTrainingInput);
    free(processedTrainingOutput);

    // free processed test data variable memory
    free(processedTestInput);
    free(processedTestOutput);

    // calculate and print training accuracy
    totalTrainingAccuracy = (totalTrainingAccuracy / (TRAINING_DATA_SETS * EPOCH)) * 100;
    printf("Training accuracy: %f%%\n", totalTrainingAccuracy);


    // print final network structure
    //printNetwork(network);

    // end timing the process
    end = clock();

    // print time taken in seconds
    cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
    printf("Time taken: %f seconds\n", cpu_time_used);

    // free training and test data variable memory
    free(trainingInput);
    free(trainingOutput);
    free(testInput);
    free(testOutput);

    // ask user if they would like to save the network
    char save = '\0';

    // consume the newline character
    while ((save = getchar()) != '\n');

    // ask user if they would like to save the network
    do {
        printf("Would you like to save the network? (y/n): ");
        char save;
        scanf("%c", &save);
        if (save == 'y') {
            printf("Saving network...\n");
            exportNetworkJSON(network, "network.json");
            printf("Network saved\n");
            break;
        } else if (save == 'n') {
            printf("Network not saved\n");
            break;
        } else {
            printf("Invalid input\n");
        }
    } while (save != 'y' && save != 'n');

    // predict user input from userInput.csv and define label to jump to
    // i can't define a variable in a label for some reason so keep it before label
    float userInput[INPUT_NODES];
    testWithUserInput:
    printf("Testing network with user input...\n");
    while (1) {
        // attempt to open userInput.csv file
        fp = fopen("userInput.csv", "r");
        if (fp == NULL) {
            perror("Error opening file");
            return 1;
        }

        // retry if file is empty
        if (getc(fp) == EOF) {
            fclose(fp);
            continue;
        }

        // reset file pointer to beginning of file
        fseek(fp, 0, SEEK_SET);

        // otherwise, read the user input and normalize it to be between 0 and 1
        printf("\n\n");
        for (int currentInputNode = 0; currentInputNode < INPUT_NODES; currentInputNode++) {
            // read the pixel values
            if (currentInputNode == INPUT_NODES - 1) {
                fscanf(fp, "%f\n", &userInput[currentInputNode]);
            } else {
                fscanf(fp, "%f,", &userInput[currentInputNode]);
            }

            // draw the user input in the terminal
            if (userInput[currentInputNode] > 0) {
                printf("X");
            } else {
                printf(" ");
            }
            if ((currentInputNode + 1) % 28 == 0) {
                printf("\n");
            }

            // normalize the pixel values to be between 0 and 1
            userInput[currentInputNode] /= 255;
            
        }
        printf("\n\n");
        fclose(fp);

        // feed the user input through the network and print the most likely number
        feedForward(network, userInput, HIDDEN_ACTIVATION, OUTPUT_ACTIVATION);
        int maxIndex = -1;
        for (int currentOutputNode = 0; currentOutputNode < OUTPUT_NODES; currentOutputNode++) {
            if (network->layers[network->layerCount - 1].values[currentOutputNode] > network->layers[network->layerCount - 1].values[maxIndex]) {
                maxIndex = currentOutputNode;
            }
            printf("Output[%d]: %f\n", currentOutputNode, network->layers[network->layerCount - 1].values[currentOutputNode]);
        }
        printf("Most Likely: %d\n", maxIndex);

        // clear userInput.csv file
        fp = fopen("userInput.csv", "w");
        fclose(fp);
    }


    // free network
    printf("Freeing network\n");
    freeNetwork(network);

    return 0;
}

// clear && clear && gcc main.c neuralNetworkLib/neuralNetwork.c -Wall -Wextra -o main && ./main
// python3 graph.py
// python3 draw.py


// Define thread functions
void *trainOnBatch(void *args) {
    // get thread parameters
    ThreadArgs threadArgs = *(ThreadArgs *)args;
    int epoch = threadArgs.epoch;
    int batchNum = threadArgs.batchNum;
    int batchSize = threadArgs.batchSize;
    float (*batchDataInput)[INPUT_NODES] = (float (*)[INPUT_NODES])threadArgs.batchDataInputPtr;
    float (*batchDataOutput)[OUTPUT_NODES] = (float (*)[OUTPUT_NODES])threadArgs.batchDataOutputPtr;

    // local vars
    FILE *fp;
    float trainingAccuracy = 0;
    int mostLikely = 0;
    int actualOutput = 0;
    float cost = 0;

    for (int currentDataSet = 0; currentDataSet < batchSize; currentDataSet++) {
        // lock mutex
        pthread_mutex_lock(&lock);

        // feed forward and back propagate the entire training data set
        feedForward(network, &batchDataInput[currentDataSet][0], HIDDEN_ACTIVATION, OUTPUT_ACTIVATION);
        backPropagate(network, &batchDataOutput[currentDataSet][0], LEARNING_RATE, HIDDEN_ACTIVATION, OUTPUT_ACTIVATION);

        // calculate cost with MSE
        cost = 0;
        for (int currentOutputNode = 0; currentOutputNode < OUTPUT_NODES; currentOutputNode++) {
            cost += pow(batchDataOutput[currentDataSet][currentOutputNode] - network->layers[network->layerCount - 1].values[currentOutputNode], 2);
        }
        cost /= OUTPUT_NODES;

        // check if the network gets the most likely output correct
        mostLikely = 0;
        actualOutput = 0;
        for (int currentOutputNode = 0; currentOutputNode < OUTPUT_NODES; currentOutputNode++) {
            if (network->layers[network->layerCount - 1].values[currentOutputNode] > network->layers[network->layerCount - 1].values[mostLikely]) {
                mostLikely = currentOutputNode;
            }
            if (batchDataOutput[currentDataSet][currentOutputNode] == 1) {
                actualOutput = currentOutputNode;
            }
        }

        // add to training accuracy if the network gets it correct
        if (mostLikely == actualOutput) {
            trainingAccuracy++;
        }

        // unlock mutex
        pthread_mutex_unlock(&lock);
    }

    // print batch number, epoch and training data set, each output, the prediction of the network, and the cost and accuracy
    // then display the number the network guessed
    // also update global vars and write to accuracy.csv file
    pthread_mutex_lock(&lock);

    // update training accuracy, threads completed, and totalThreadsCompleted
    totalTrainingAccuracy += trainingAccuracy;
    totalThreadsCompleted++;
    threadsCompleted++;

    // open accuracy.csv file
    if (LOG_ACCURACY) {
        fp = fopen("accuracy.csv", "a");
        if (fp == NULL) {
            perror("Error opening file");
            return NULL;
        }

        // write training accuracy to accuracy.csv file
        fprintf(fp, "Train: %f\n", (totalTrainingAccuracy / (batchSize * totalThreadsCompleted)) * 100);

        // close accuracy.csv file
        fclose(fp);
    }

    // print info
    printf("Batch #%d completed\n", batchNum);
    printf("Epoch %d/%d, Training Data Set %d\n", epoch+1, EPOCH, batchSize);
    for (int i = 0; i < OUTPUT_NODES; i++) {
        printf("Output[%d]: %f, Expected: %f\n", i, network->layers[network->layerCount - 1].values[i], batchDataOutput[batchSize-1][i]);
    }
    printf("Most Likely: %d, Actual Value: %d\n", mostLikely, actualOutput);
    printf("Cost: %f, Current Training Accuracy: %f%%\n", cost, (totalTrainingAccuracy / (batchSize * totalThreadsCompleted)) * 100);
    printf("\n");
    displayImage(&batchDataInput[batchSize-1][0], (int)sqrt(INPUT_NODES), (int)sqrt(INPUT_NODES));
    printf("\n%f%% Finished Epoch %d\n", (((float)threadsCompleted / (TRAINING_DATA_SETS / MINI_BATCH_SIZE)) * 100), epoch+1);
    printf("\n");

    // unlock mutex
    pthread_mutex_unlock(&lock);

    // exit thread
    return NULL;
}


// Define network functions

// Test the current network against the testing data
void testNetwork(float *inputData, int inputDataSize, float *outputData, int outputDataSize, int dataLength) {
    // pre-process test data
    printf("Pre-Processing test data\n");

    for (int currentDataSet = 0; currentDataSet < dataLength; currentDataSet++) {
        // randomly scale the image between 0.8 and 1.2
        float scale = (float)rand() / RAND_MAX * 0.4 + 0.8;
        scaleData(&inputData[currentDataSet * inputDataSize], (int)sqrt(INPUT_NODES), (int)sqrt(INPUT_NODES), scale);

        // randomly rotate the image between -15 and 15 degrees
        float angle = (float)rand() / RAND_MAX * 30 - 15;
        rotateData(&inputData[currentDataSet * inputDataSize], (int)sqrt(INPUT_NODES), (int)sqrt(INPUT_NODES), angle);
        
        // randomly translate the image between -2 and 2 pixels
        int offsetX = rand() % 4 - 2;
        int offsetY = rand() % 4 - 2;
        translateData(&inputData[currentDataSet * inputDataSize], (int)sqrt(INPUT_NODES), (int)sqrt(INPUT_NODES), offsetX, offsetY);

        // add noise to the current image
        addNoise(&inputData[currentDataSet * inputDataSize], INPUT_NODES, 1);
    }

    printf("Testing network...\n\n");

    // feed values through the network and check if the most likely output is correct
    for (int currentDataSet = 0; currentDataSet < dataLength; currentDataSet++) {
        //printf("Testing data set %d/%d\n", currentDataSet+1, TESTING_DATA_SETS);
        feedForward(network, &inputData[currentDataSet * inputDataSize], HIDDEN_ACTIVATION, OUTPUT_ACTIVATION);
        int maxIndex = 0;
        for (int currentOutputNode = 0; currentOutputNode < outputDataSize; currentOutputNode++) {
            if (network->layers[network->layerCount - 1].values[currentOutputNode] > network->layers[network->layerCount - 1].values[maxIndex]) {
                maxIndex = currentOutputNode;
            }
        }
        if (outputData[currentDataSet * outputDataSize + maxIndex] == 1) {
            testCorrect++;
        }

        // print every 5% (0.05) of the data set (dataLength * 0.05)
        if (currentDataSet % (int)((float)dataLength * 0.05) == 0) {
            printf("%d%% Finished\n", (int)(((float)(currentDataSet) / TESTING_DATA_SETS) * 100));
        }
    }

    if (LOG_ACCURACY) {
        // open accuracy.csv file
        FILE *fp = fopen("accuracy.csv", "a");
        if (fp == NULL) {
            perror("Error opening file");
            return;
        }

        // write testing accuracy to accuracy.csv file
        fprintf(fp, "Test: %f\n", (float)testCorrect / TESTING_DATA_SETS * 100);

        // close accuracy.csv file
        fclose(fp);
    }
    
    printf("100%% Finished\n");
}

// Shuffle input data but keep input and output data equvelent
void shuffleData(float *inputData, int inputDataSize, float *outputData, int outputDataSize, int dataLength) {
    for (int i = 0; i < dataLength; i++) {
        int randomIndex = rand() % dataLength;
        float tempInput[inputDataSize];
        float tempOutput[outputDataSize];
        for (int j = 0; j < INPUT_NODES; j++) {
            tempInput[j] = inputData[i * inputDataSize + j];
        }
        for (int j = 0; j < outputDataSize; j++) {
            tempOutput[j] = outputData[i * outputDataSize + j];
        }
        for (int j = 0; j < inputDataSize; j++) {
            inputData[i * inputDataSize + j] = inputData[randomIndex * inputDataSize + j];
        }
        for (int j = 0; j < outputDataSize; j++) {
            outputData[i * outputDataSize + j] = outputData[randomIndex * outputDataSize + j];
        }
        for (int j = 0; j < inputDataSize; j++) {
            inputData[randomIndex *inputDataSize + j] = tempInput[j];
        }
        for (int j = 0; j < outputDataSize; j++) {
            outputData[randomIndex * outputDataSize + j] = tempOutput[j];
        }
    }
}

// noise function to add random noise to the images
void addNoise(float *inputData, int dataLength, int noiseLevel) {
    for (int i = 0; i < dataLength; i++) {
        if (rand() % 100 < noiseLevel) {
            inputData[i] = (float)rand() / RAND_MAX;
        }
    }
}

// translate the image by the given offsets
void translateData(float *inputData, int dataWidth, int dataHeight, int offsetX, int offsetY) {
    float tempData[dataWidth * dataHeight];
    for (int i = 0; i < dataWidth * dataHeight; i++) {
        tempData[i] = inputData[i];
    }
    for (int i = 0; i < dataWidth * dataHeight; i++) {
        int x = i % dataWidth;
        int y = i / dataWidth;
        int newX = x + offsetX;
        int newY = y + offsetY;
        if (newX >= 0 && newX < dataWidth && newY >= 0 && newY < dataHeight) {
            inputData[newY * dataWidth + newX] = tempData[i];
        } else {
            inputData[i] = 0;
        }
    }
}

// scale the image by the given scale while keeping the image size the same (if the image is 28x28, the image will still be 28x28)
void scaleData(float *inputData, int dataWidth, int dataHeight, float scale) {
    float tempData[dataWidth * dataHeight];
    for (int imageSize = 0; imageSize < dataWidth * dataHeight; imageSize++) {
        tempData[imageSize] = inputData[imageSize];
    }
    for (int width = 0; width < dataWidth; width++) {
        for (int height = 0; height < dataHeight; height++) {
            int nearestX = (int)(width * scale + 0.5);
            int nearestY = (int)(height * scale + 0.5);
            if (nearestX >= 0 && nearestX < dataWidth && nearestY >= 0 && nearestY < dataHeight) {
                inputData[height * dataWidth + width] = tempData[nearestY * dataWidth + nearestX];
            } else {
                inputData[height * dataWidth + width] = 0;
            }
        }
    }
}

// rotate the image by any given angle in degrees
void rotateData(float *inputData, int dataWidth, int dataHeight, float angle) {
    float tempData[dataWidth * dataHeight];
    for (int imageSize = 0; imageSize < dataWidth * dataHeight; imageSize++) {
        tempData[imageSize] = inputData[imageSize];
    }

    float centerX = dataWidth / 2;
    float centerY = dataHeight / 2;
    float radians = angle * M_PI / 180;
    for (int width = 0; width < dataWidth; width++) {
        for (int height = 0; height < dataHeight; height++) {
            float x = cos(radians) * (width - centerX) - sin(radians) * (height - centerY) + centerX;
            float y = sin(radians) * (width - centerX) + cos(radians) * (height - centerY) + centerY;
            if (x >= 0 && x < dataWidth && y >= 0 && y < dataHeight) {
                inputData[height * dataWidth + width] = tempData[(int)y * dataWidth + (int)x];
            } else {
                inputData[height * dataWidth + width] = 0;
            }
        }
    }
}

// display image in terminal
void displayImage(float *inputData, int dataWidth, int dataHeight) {
    for (int i = 0; i < dataWidth * dataHeight; i++) {
        if (inputData[i] > 0.5) {
            printf("X");
        } else if (inputData[i] > 0) {
            printf("x");
        } else {
            printf(" ");
        }
        if ((i + 1) % dataWidth == 0) {
            printf("\n");
        }
    }
}
