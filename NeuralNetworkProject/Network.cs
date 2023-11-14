namespace NeuralNetworkProject;

public class Network
{
    private Random _random = new();
    private List<double[]> _neurons = new();
    private List<double[]> _biases = new();
    private List<double[,]> _weights = new();

    private double _learningRate = 0.5;

    public void AddLayer(int newNeuronsCount) {
        _neurons.Add(new double[newNeuronsCount]);

        Console.WriteLine("Added " + newNeuronsCount + " neurons");

        if (GetLayerCount() > 1) {
            _biases.Add(new double[newNeuronsCount]);
            Console.WriteLine("\tAdded " + newNeuronsCount + " biases");

            _weights.Add(new double[newNeuronsCount, GetNeuronLayer(-2).Length]);

            for (var i = 0; i < newNeuronsCount; i++)
            for (var j = 0; j < GetNeuronLayer(-2).Length; j++)
                _weights[^1][i, j] = InitialWeight();

            Console.WriteLine("\tAdded " + newNeuronsCount + "x" + GetNeuronLayer(-2).Length + " weights");
        }
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
        return _random.NextDouble();
    }

    public void Train(double[] trainingInputs, double[] trainingOutputs) {
        // forward pass
        {
            // assign the inputs into the input layer
            var inputLayer = GetNeuronLayer(0);
            for (var i = 0; i < inputLayer.Length; i++) inputLayer[i] = trainingInputs[i];
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


                GetNeuronLayer(layerIndex)[j] =
                    Sigmoid.Func(innerPotential);
            }
        }
    }

    public void PrintState() {
        Console.WriteLine("Neurons:");
        for (var i = 0; i < GetLayerCount(); i++) {
            Console.Write("\tLayer " + i + ": ");
            for (var j = 0; j < GetNeuronLayer(i).Length; j++) Console.Write(Math.Round(GetNeuronLayer(i)[j], 3) + " ");

            Console.WriteLine();
        }

        Console.WriteLine("Biases:");
        for (var i = 0; i < GetLayerCount() - 1; i++) {
            Console.Write("\tLayer " + i + ": ");
            for (var j = 0; j < GetBiasLayer(i).Length; j++) Console.Write(Math.Round(GetBiasLayer(i)[j], 3) + " ");

            Console.WriteLine();
        }

        Console.WriteLine("Weights:");
        for (var i = 0; i < GetLayerCount() - 1; i++) {
            Console.WriteLine("\tLayer " + i + ":");

            for (var j = 0; j < GetWeightLayer(i).GetLength(0); j++) {
                Console.Write("\t\t");
                for (var k = 0; k < GetWeightLayer(i).GetLength(1); k++)
                    Console.Write(Math.Round(GetWeightLayer(i)[j, k], 3) + " ");

                Console.WriteLine();
            }

            Console.WriteLine();
        }
    }
}