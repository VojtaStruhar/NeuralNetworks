// See https://aka.ms/new-console-template for more information

using System.Text;
using NeuralNetworkProject;

// var colors = CsvUtils.ReadColors("./data/fashion_mnist_train_vectors.csv");
// Console.WriteLine("Colors: " + colors.Length + "x" + colors[0].Length);
// for (var i = 0; i < 16; i++) Console.WriteLine(colors[14][i]);
//
// var numericLabels = CsvUtils.ReadLabels("./data/fashion_mnist_train_labels.csv");
// Console.WriteLine("Labels: " + numericLabels.Length);
//
// var trainOutputs = Utils.OneHotEncode(numericLabels, 10);
// for (var i = 0; i < 16; i++) {
//     Console.Write(numericLabels[i] + " - ");
//     for (var j = 0; j < trainOutputs[i].Length; j++) Console.Write(trainOutputs[i][j] + " ");
//     Console.WriteLine();
// }
//
// return;

const int NUMBER_OF_INPUTS = 2;
const int NUMBER_OF_OUTPUTS = 1;

var trainingInputs = CsvUtils.ReadVectors("./data/xor_train_inputs.csv");
var trainingOutputs = CsvUtils.ReadVectors("./data/xor_train_outputs.csv");


var network = new Network();

network.AddLayer(NUMBER_OF_INPUTS, ActivationFunction.Sigmoid);
network.AddLayer(4, ActivationFunction.Sigmoid);
network.AddLayer(NUMBER_OF_OUTPUTS, ActivationFunction.Sigmoid);

network.TrainEpochs(trainingInputs, trainingOutputs, 10);

network.PrintState();

Console.WriteLine("Testing the neural network...");
var testInputs = CsvUtils.ReadVectors("./data/xor_test_inputs.csv");
var correctOutputs = CsvUtils.ReadVectors("./data/xor_test_outputs.csv");


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