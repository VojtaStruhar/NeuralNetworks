namespace NeuralNetworkProject;

public class Network
{
    private Random _random = new();
    private List<double[]> _neurons = new();
    private List<double[]> _biases = new();
    private List<double[,]> _weights = new();
    private List<ActivationFunction> _activationFunctions = new();

    private double _learningRate = 0.2;
    private double _initialWeightMultiplier = 1.0;

    public Network AddLayer(int newNeuronsCount, ActivationFunction activationFunction) {
        _neurons.Add(new double[newNeuronsCount]);
        _activationFunctions.Add(activationFunction);

        if (GetLayerCount() > 1) {
            _biases.Add(new double[newNeuronsCount]);

            _weights.Add(new double[newNeuronsCount, GetNeuronLayer(-2).Length]);

            for (var i = 0; i < newNeuronsCount; i++)
            for (var j = 0; j < GetNeuronLayer(-2).Length; j++)
                _weights[^1][i, j] = InitialWeight();
        }

        return this;
    }

    public Network SetLearningRate(double lr) {
        _learningRate = lr;
        return this;
    }

    public Network SetInitialWeightMultiplier(double multiplier) {
        _initialWeightMultiplier = multiplier;
        return this;
    }

    private int GetLayerCount() {
        return _neurons.Count;
    }

    private double[] GetNeuronLayer(int layer) {
        if (layer < 0) return _neurons[^Math.Abs(layer)];
        return _neurons[layer];
    }

    private double[] GetBiasLayer(int layer) {
        if (layer < 0) return _biases[^Math.Abs(layer)];
        return _biases[layer];
    }

    private double[,] GetWeightLayer(int layer) {
        if (layer < 0) return _weights[^Math.Abs(layer)];
        return _weights[layer];
    }


    private double InitialWeight() {
        // This returns a number in [0,1)
        // TODO: I probably want this smaller
        return _random.NextDouble() * _initialWeightMultiplier;
    }


    public void TrainEpochs(double[][] trainingInputs, double[][] trainingOutputs, int numberOfEpochs) {
        var trainingInputsOrder = new int[trainingInputs.Length];
        for (var i = 0; i < trainingInputsOrder.Length; i++) trainingInputsOrder[i] = i;

        for (var i = 0; i < numberOfEpochs; i++) {
            if (i % (numberOfEpochs / 10) == 0)
                Console.Write("Epoch " + i + "...\r");

            Utils.Shuffle(_random, trainingInputsOrder);
            for (var j = 0; j < trainingInputs.Length; j++) {
                var trainingIndex = trainingInputsOrder[j];

                Train(trainingInputs[trainingIndex], trainingOutputs[trainingIndex]);
            }
        }

        Console.WriteLine("Trained " + numberOfEpochs + " epochs!");
    }

    public void Train(double[] trainingInputs, double[] trainingOutputs) {
        {
            // assign the inputs into the input layer
            var inputLayer = GetNeuronLayer(0);
            for (var i = 0; i < inputLayer.Length; i++) inputLayer[i] = trainingInputs[i];
        }

        /* ------------------------- *
         *       FORWARD PASS        *
         * ------------------------- */

        // compute hidden layer activation
        for (var layerIndex = 1; layerIndex < GetLayerCount(); layerIndex++) {
            var currentNeurons = GetNeuronLayer(layerIndex);
            var currentBiases = GetBiasLayer(layerIndex - 1); // There is 1 layer of biases less than neurons
            var currentWeights = GetWeightLayer(layerIndex - 1);
            var previousNeurons = GetNeuronLayer(layerIndex - 1);

            for (var j = 0; j < currentNeurons.Length; j++) {
                var innerPotential = currentBiases[j];

                // safe to do -1, because Im starting from 1
                for (var k = 0; k < previousNeurons.Length; k++)
                    innerPotential += previousNeurons[k] * currentWeights[j, k];


                currentNeurons[j] = _activationFunctions[layerIndex].Func(innerPotential);
            }
        }

        /* ------------------------- *
         *      BACKPROPAGATION      *
         * ------------------------- */


        // Go from the back to the front
        double[] previousDeltas;

        {
            // I need a starting point deltas - outputs
            var outputLayer = GetNeuronLayer(-1);

            var deltas = new double[outputLayer.Length];
            for (var i = 0; i < deltas.Length; i++) {
                var error = trainingOutputs[i] - outputLayer[i];
                deltas[i] = error * _activationFunctions[^1].Derivative(outputLayer[i]);
            }

            // apply the changes of output weights
            for (var i = 0; i < outputLayer.Length; i++) {
                GetBiasLayer(-1)[i] += _learningRate * deltas[i];
                var outputWeights = GetWeightLayer(-1);
                var nextHiddlenLayer = GetNeuronLayer(-2);

                for (var j = 0; j < nextHiddlenLayer.Length; j++)
                    outputWeights[i, j] += _learningRate * deltas[i] * nextHiddlenLayer[j];
            }

            previousDeltas = deltas;
        }

        // Now compute the rest of the hidden layers
        for (var layerIndex = GetLayerCount() - 2; layerIndex >= 1; layerIndex--) {
            var currentNeurons = GetNeuronLayer(layerIndex);
            var currentBiases = GetBiasLayer(layerIndex - 1);
            var currentWeights = GetWeightLayer(layerIndex - 1);
            var currentDeltas = new double[currentNeurons.Length];

            var nextNeurons = GetNeuronLayer(layerIndex - 1);
            var previousWeights = GetWeightLayer(layerIndex);

            for (var i = 0; i < currentNeurons.Length; i++) {
                var error = 0.0;
                for (var j = 0; j < previousDeltas.Length; j++) error += previousDeltas[j] * previousWeights[j, i];

                currentDeltas[i] = error * _activationFunctions[layerIndex].Derivative(currentNeurons[i]);
            }

            // apply the weight changes
            for (var i = 0; i < currentNeurons.Length; i++) {
                currentBiases[i] += _learningRate * currentDeltas[i];

                for (var j = 0; j < nextNeurons.Length; j++)
                    currentWeights[i, j] += _learningRate * currentDeltas[i] * nextNeurons[j];
            }

            previousDeltas = currentDeltas;
        }
    }

    public double[] Predict(double[] inputs) {
        {
            // assign the inputs into the input layer
            var inputLayer = GetNeuronLayer(0);
            for (var i = 0; i < inputLayer.Length; i++) inputLayer[i] = inputs[i];
        }

        // compute hidden layer activation
        for (var layerIndex = 1; layerIndex < GetLayerCount(); layerIndex++) {
            // start with bias

            var currentNeurons = GetNeuronLayer(layerIndex);
            var currentBiases = GetBiasLayer(layerIndex - 1); // There is 1 layer of biases less than neurons
            var currentWeights = GetWeightLayer(layerIndex - 1);
            var previousNeurons = GetNeuronLayer(layerIndex - 1);

            for (var j = 0; j < currentNeurons.Length; j++) {
                var innerPotential = currentBiases[j];

                // safe to do -1, because Im starting from 1
                for (var k = 0; k < previousNeurons.Length; k++)
                    innerPotential += previousNeurons[k] * currentWeights[j, k];


                GetNeuronLayer(layerIndex)[j] = _activationFunctions[layerIndex].Func(innerPotential);
            }
        }

        var outputLayer = GetNeuronLayer(-1);
        var output = new double[outputLayer.Length];
        for (var i = 0; i < outputLayer.Length; i++) output[i] = outputLayer[i];

        return output;
    }

    public void PrintState() {
        for (var i = 0; i < GetLayerCount(); i++) {
            Console.WriteLine("Neurons " + i + ": \t" + Utils.FormatArray(GetNeuronLayer(i), 16));
            if (i > 0) Console.WriteLine("Biases for " + i + ":\t" + Utils.FormatArray(GetBiasLayer(i - 1), 16));

            Console.WriteLine();
        }

        // Console.WriteLine("Weights:");
        // for (var i = 0; i < GetLayerCount() - 1; i++) {
        //     Console.WriteLine("\tLayer " + i + ":");
        //
        //     for (var j = 0; j < GetWeightLayer(i).GetLength(0); j++) {
        //         Console.Write("\t\t");
        //         for (var k = 0; k < GetWeightLayer(i).GetLength(1); k++)
        //             Console.Write(Math.Round(GetWeightLayer(i)[j, k], 3) + " ");
        //
        //         Console.WriteLine();
        //     }
        //
        //     Console.WriteLine();
        // }
    }
}