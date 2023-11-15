// See https://aka.ms/new-console-template for more information

using NeuralNetworkProject;

const int NUMBER_OF_INPUTS = 2;
const int NUMBER_OF_OUTPUTS = 1;
const int NUMBER_OF_TRAINING_VECTORS = 100;

var numberOfEpochs = 5000;


var trainingInputs = new double[NUMBER_OF_TRAINING_VECTORS][];
var trainingOutputs = new double[NUMBER_OF_TRAINING_VECTORS][];

var r = new Random();
for (var i = 0; i < NUMBER_OF_TRAINING_VECTORS; i++) {
    var a = r.Next() % 2;
    var b = r.Next() % 2;
    trainingInputs[i] = new double[NUMBER_OF_INPUTS] { a, b };
    trainingOutputs[i] = new double[NUMBER_OF_OUTPUTS] { a ^ b };
}


var network = new Network();

network.AddLayer(NUMBER_OF_INPUTS, ActivationFunction.Sigmoid);
network.AddLayer(4, ActivationFunction.Sigmoid);
network.AddLayer(NUMBER_OF_OUTPUTS, ActivationFunction.Sigmoid);


var trainingInputsOrder = new int[NUMBER_OF_TRAINING_VECTORS];
for (var i = 0; i < trainingInputsOrder.Length; i++) trainingInputsOrder[i] = i;

for (var i = 0; i < numberOfEpochs; i++) {
    Utils.Shuffle(r, trainingInputsOrder);
    for (var j = 0; j < trainingInputs.Length; j++) {
        var trainingIndex = trainingInputsOrder[j];
        network.Train(trainingInputs[trainingIndex], trainingOutputs[trainingIndex]);
    }
}


network.PrintState();

Console.WriteLine("Testing the neural network...");
var testInputs = new double[4][];
testInputs[0] = new double[] { 0, 0 };
testInputs[1] = new double[] { 0, 1 };
testInputs[2] = new double[] { 1, 0 };
testInputs[3] = new double[] { 1, 1 };

var correctOutputs = new double[4] { 0, 1, 1, 0 };

for (var i = 0; i < testInputs.Length; i++) {
    var output = network.Predict(testInputs[i]);
    var errorPercentage = (int)(double.Abs(output[0] - correctOutputs[i]) * 100);
    Console.WriteLine("In: " + testInputs[i][0] + "," + testInputs[i][1] +
                      "\t| Output: " + output[0] +
                      "\t| Expected: " + correctOutputs[i] +
                      "\t| Error: " + +errorPercentage + "%");
}