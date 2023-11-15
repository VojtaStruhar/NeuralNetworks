// See https://aka.ms/new-console-template for more information

using NeuralNetworkProject;

Console.WriteLine("Reading the training data...");

var trainingInputs = CsvUtils.ReadColors("./data/fashion_mnist_train_vectors.csv", 100);

var trainingOutputs = CsvUtils.ReadAndEncodeLabels("./data/fashion_mnist_train_labels.csv", 10, 100);


const int NUMBER_OF_INPUTS = 28 * 28;
const int NUMBER_OF_OUTPUTS = 10;

Console.WriteLine("Creating the neural network...");

var network = new Network();

network.AddLayer(NUMBER_OF_INPUTS, ActivationFunction.Sigmoid);
network.AddLayer(64, ActivationFunction.Sigmoid);
network.AddLayer(32, ActivationFunction.Sigmoid);
network.AddLayer(16, ActivationFunction.Sigmoid);
network.AddLayer(NUMBER_OF_OUTPUTS, ActivationFunction.Sigmoid);

Console.WriteLine("Training...");

network.TrainEpochs(trainingInputs, trainingOutputs, 10);


Console.WriteLine("Testing the neural network...");
var testInputs = CsvUtils.ReadColors("./data/fashion_mnist_test_vectors.csv", 100);
var correctLabels = CsvUtils.ReadLabels("./data/fashion_mnist_test_labels.csv", 100);
var correctOutputs = correctLabels.Select(label => Utils.OneHotEncode(label, 10)).ToArray();

var correctPredictions = 0;

for (var i = 0; i < testInputs.Length; i++) {
    var inputs = testInputs[i];
    var networkOutput = network.Predict(inputs);
    var correctOutput = correctOutputs[i];

    var guessedLabel = Utils.OneHotDecode(networkOutput);
    if ((int)correctLabels[i] == guessedLabel) correctPredictions++;

    var errorAverage = 0.0;
    {
        var errorSum = 0.0;
        for (var j = 0; j < networkOutput.Length; j++) errorSum += double.Abs(networkOutput[j] - correctOutput[j]);
        errorAverage = errorSum / networkOutput.Length;
    }


    var errorPercentage = (int)(errorAverage * 100);
    Console.WriteLine("Test " + i +
                      "\t| Output: " + Utils.FormatArray(networkOutput, 4) +
                      "\t| Error: " + +errorPercentage + "%" +
                      "\t| Label: " + guessedLabel +
                      "\t| Expected: " + correctLabels[i] +
                      (guessedLabel == (int)correctLabels[i] ? " (yay)" : "")
    );
}

Console.WriteLine("\nGot " + correctPredictions + " correct predictions right out of " + testInputs.Length +
                  " tests. That's " + Math.Round(correctPredictions / (double)testInputs.Length * 100, 1) + "%");