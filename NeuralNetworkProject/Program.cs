// See https://aka.ms/new-console-template for more information

using NeuralNetworkProject;

Console.WriteLine("Hello, World!");

const int NUMBER_OF_INPUTS = 2;
const int NUMBER_OF_HIDDEN_NEURONS = 2; // this is assuming just a single hidden layer
const int NUMBER_OF_OUTPUTS = 1;
const int NUMBER_OF_TRAINING_VECTORS = 100;

void Shuffle<T>(Random rng, T[] array) {
    var n = array.Length;
    while (n > 1) {
        var k = rng.Next(n--);
        var temp = array[n];
        array[n] = array[k];
        array[k] = temp;
    }
}

double Sigmoid(double x) {
    return 1 / (1 + double.Exp(-x));
}

double SigmoidDerivative(double x) {
    return x * (1 - x);
}

double initialWeight() {
    // This returns a number in [0,1)
    // TODO: I probably want this smaller
    return new Random().NextDouble();
}

var learningRate = 0.1;

var hiddenLayer = new double[NUMBER_OF_HIDDEN_NEURONS];
var outputLayer = new double[NUMBER_OF_OUTPUTS];

var hiddenLayerBias = new double[NUMBER_OF_HIDDEN_NEURONS];
var outputLayerBias = new double[NUMBER_OF_OUTPUTS];

var hiddenWeights = new double[NUMBER_OF_INPUTS, NUMBER_OF_HIDDEN_NEURONS];
var outputWeights = new double[NUMBER_OF_HIDDEN_NEURONS, NUMBER_OF_OUTPUTS];

var trainingInputs = new double[NUMBER_OF_TRAINING_VECTORS, NUMBER_OF_INPUTS];
var trainingOutputs = new double[NUMBER_OF_TRAINING_VECTORS, NUMBER_OF_OUTPUTS];
var r = new Random();
for (var i = 0; i < NUMBER_OF_TRAINING_VECTORS; i++) {
    var a = r.Next() % 2;
    var b = r.Next() % 2;
    trainingInputs[i, 0] = a;
    trainingInputs[i, 1] = b;
    trainingOutputs[i, 0] = a ^ b;
}

for (var i = 0; i < NUMBER_OF_INPUTS; i++)
for (var j = 0; j < NUMBER_OF_HIDDEN_NEURONS; j++)
    hiddenWeights[i, j] = initialWeight();

for (var i = 0; i < NUMBER_OF_HIDDEN_NEURONS; i++)
for (var j = 0; j < NUMBER_OF_OUTPUTS; j++)
    outputWeights[i, j] = initialWeight();

for (var i = 0; i < NUMBER_OF_OUTPUTS; i++)
    outputLayerBias[i] = initialWeight();


var orderOfTrainingInputs = new int[NUMBER_OF_TRAINING_VECTORS];
for (var i = 0; i < NUMBER_OF_TRAINING_VECTORS; i++) orderOfTrainingInputs[i] = i;

var numberOfEpochs = 10_000;
// Train the neural network for a number of epochs
for (var epoch = 0; epoch < numberOfEpochs; epoch++) {
    // training loop
    Shuffle(new Random(), orderOfTrainingInputs);

    for (var x = 0; x < NUMBER_OF_TRAINING_VECTORS; x++) {
        var i = orderOfTrainingInputs[x];

        // forward pass


        // compute hidden layer activation
        for (var j = 0; j < NUMBER_OF_HIDDEN_NEURONS; j++) {
            // start with bias
            var innerPotential = hiddenLayerBias[j];

            // Here it's number of inputs, because we only have one hidden layer, so it's clear which one is the previous one :)
            for (var k = 0; k < NUMBER_OF_INPUTS; k++)
                innerPotential += trainingInputs[i, k] * hiddenWeights[k, j];

            hiddenLayer[j] = Sigmoid(innerPotential); // near 1 if it's activated, 0 if it's not
        }

        // compute output layer activation
        for (var j = 0; j < NUMBER_OF_OUTPUTS; j++) {
            // start with bias
            var innerPotential = hiddenLayerBias[j];

            for (var k = 0; k < NUMBER_OF_HIDDEN_NEURONS; k++) // now the previous is the one hidden layer
                innerPotential += hiddenLayer[k] * outputWeights[k, j];

            outputLayer[j] = Sigmoid(innerPotential);
        }

        // Console.WriteLine("Output: " + outputLayer[0] +
        //                   "\t| Expected: " +
        //                   trainingOutputs[i, 0] + "\t| Error: " + new string('-',
        //                       (int)double.Round(double.Abs(outputLayer[0] - trainingInputs[i, 0]) * 100)));

        // Backpropagation

        // Copmute the change in output weights
        var outputWeightDeltas = new double[NUMBER_OF_OUTPUTS];
        for (var j = 0; j < NUMBER_OF_OUTPUTS; j++) {
            var error = trainingOutputs[i, j] - outputLayer[j];
            var delta = error * SigmoidDerivative(outputLayer[j]);

            outputWeightDeltas[j] = delta;
        }

        // Compute the change in hidden weights
        var hiddenDeltas = new double[NUMBER_OF_HIDDEN_NEURONS];
        for (var j = 0; j < NUMBER_OF_HIDDEN_NEURONS; j++) {
            var error = 0.0;
            for (var k = 0; k < NUMBER_OF_OUTPUTS; k++)
                error += outputWeightDeltas[k] * outputWeights[j, k];

            var delta = error * SigmoidDerivative(hiddenLayer[j]);
            hiddenDeltas[j] = delta;
        }

        // apply the change in output weights
        for (var j = 0; j < NUMBER_OF_OUTPUTS; j++) {
            outputLayerBias[j] += outputWeightDeltas[j] * learningRate;

            for (var k = 0; k < NUMBER_OF_HIDDEN_NEURONS; k++)
                // the changes can be positive or negative!
                outputWeights[k, j] += hiddenLayer[k] * outputWeightDeltas[j] * learningRate;
        }

        // apply the change in hidden weights
        for (var j = 0; j < NUMBER_OF_HIDDEN_NEURONS; j++) {
            hiddenLayerBias[j] += hiddenDeltas[j] * learningRate;

            for (var k = 0; k < NUMBER_OF_INPUTS; k++)
                hiddenWeights[k, j] += trainingInputs[i, k] * hiddenDeltas[j] * learningRate;
        }
    }

    Console.WriteLine("Epoch " + epoch);
}

Console.WriteLine("Training complete!");

// final hidden weights
Console.WriteLine("Hidden weights:");
for (var i = 0; i < NUMBER_OF_HIDDEN_NEURONS; i++) {
    for (var j = 0; j < NUMBER_OF_INPUTS; j++) Console.Write(hiddenWeights[j, i] + " ");
    Console.WriteLine();
}

// final output weights
Console.WriteLine("Output weights:");
for (var i = 0; i < NUMBER_OF_HIDDEN_NEURONS; i++) {
    for (var j = 0; j < NUMBER_OF_OUTPUTS; j++) Console.Write(outputWeights[i, j] + " ");
    Console.WriteLine();
}

// final hidden biases
Console.WriteLine("Hidden biases:");
for (var i = 0; i < NUMBER_OF_HIDDEN_NEURONS; i++) Console.Write(hiddenLayerBias[i] + " ");
Console.WriteLine();

// final output biases
Console.WriteLine("Output biases:");
for (var i = 0; i < NUMBER_OF_OUTPUTS; i++) Console.Write(outputLayerBias[i] + " ");
Console.WriteLine();

TestNeuralNetwork();

void TestNeuralNetwork() {
// test the neural network
// COPILOT FROM NOW ON
    Console.WriteLine("Testing the neural network...");
    var testInputs = new double[4, 2] {
        { 0, 0 },
        { 0, 1 },
        { 1, 0 },
        { 1, 1 }
    };
    var correctOutputs = new double[4] { 0, 1, 1, 0 };
    for (var i = 0; i < 4; i++) {
        // compute hidden layer activation
        for (var j = 0; j < NUMBER_OF_HIDDEN_NEURONS; j++) {
            // start with bias
            var innerPotential = hiddenLayerBias[j];

            // Here it's number of inputs, because we only have one hidden layer, so it's clear which one is the previous one :)
            for (var k = 0; k < NUMBER_OF_INPUTS; k++)
                innerPotential += testInputs[i, k] * hiddenWeights[k, j];

            hiddenLayer[j] = Sigmoid(innerPotential); // near 1 if it's activated, 0 if it's not
        }

        // compute output layer activation
        for (var j = 0; j < NUMBER_OF_OUTPUTS; j++) {
            // start with bias
            var innerPotential = hiddenLayerBias[j];

            for (var k = 0; k < NUMBER_OF_HIDDEN_NEURONS; k++) // now the previous is the one hidden layer
                innerPotential += hiddenLayer[k] * outputWeights[k, j];

            outputLayer[j] = Sigmoid(innerPotential);
        }

        var errorPercentage = (int)(double.Abs(outputLayer[0] - correctOutputs[i]) * 100);
        Console.WriteLine("Output: " + outputLayer[0] +
                          "\t| Expected: " + correctOutputs[i] +
                          "\t| Error: " + +errorPercentage + "%");
    }
}