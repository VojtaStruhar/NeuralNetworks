// See https://aka.ms/new-console-template for more information

using NeuralNetworkProject;

Console.WriteLine("Reading the training data...");

var trainingInputs = CsvUtils.ReadColors("./data/fashion_mnist_train_vectors.csv");

var labels = CsvUtils.ReadLabels("./data/fashion_mnist_train_labels.csv");
var trainingOutputs = Utils.OneHotEncode(labels, 10);


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

network.TrainEpochs(trainingInputs.Take(100).ToArray(), trainingOutputs.Take(100).ToArray(), 10);


Console.WriteLine("Testing the neural network...");
var testInputs = CsvUtils.ReadColors("./data/fashion_mnist_test_vectors.csv");
var correctOutputs = Utils.OneHotEncode(CsvUtils.ReadLabels("./data/fashion_mnist_test_labels.csv"), 10);


for (var i = 0; i < Math.Min(testInputs.Length, 10); i++) {
    var networkOutput = network.Predict(testInputs[i]);
    var correctOutput = correctOutputs[i];

    var errorAverage = 0.0;
    {
        var errorSum = 0.0;
        for (var j = 0; j < networkOutput.Length; j++) errorSum += double.Abs(networkOutput[j] - correctOutput[j]);
        errorAverage = errorSum / networkOutput.Length;
    }


    var errorPercentage = (int)(errorAverage * 100);
    Console.WriteLine("Test " + i +
                      "\t| In: " + Utils.FormatArray(testInputs[i]) +
                      "\t| Output: " + Utils.FormatArray(networkOutput) +
                      "\t| Expected: " + Utils.FormatArray(correctOutputs[i]) +
                      "\t| Error: " + +errorPercentage + "%");
}