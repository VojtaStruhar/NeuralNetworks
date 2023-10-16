namespace NeuralNetworkProject;

public class Network
{
    private InputNeuron[] inputLayer;
    private INeuron[][] hiddenLayers;
    private INeuron[] outputLayer;

    public Network(int inputNeurons, int[] hiddenNeurons, int outputNeurons) {
        inputLayer = new InputNeuron[inputNeurons];
        for (var i = 0; i < inputLayer.Length; i++) inputLayer[i] = new InputNeuron(0f);

        hiddenLayers = new INeuron[hiddenNeurons.Length][];
        for (var layerIndex = 0; layerIndex < hiddenNeurons.Length; layerIndex++) {
            var layer = new INeuron[hiddenNeurons[layerIndex]];
            hiddenLayers[layerIndex] = layer;

            // This gets global layer with index - 0 is input
            var previousLayer = GetLayer(layerIndex);

            for (var j = 0; j < hiddenNeurons[layerIndex]; j++) layer[j] = new HiddenNeuron(previousLayer);
        }

        outputLayer = new HiddenNeuron[outputNeurons];
        for (var i = 0; i < outputNeurons; i++)
            outputLayer[i] = new HiddenNeuron(GetLayer(hiddenLayers.Length));
    }

    private INeuron[] GetLayer(int index) {
        if (index == 0) return inputLayer;

        if (index < hiddenLayers.Length) return hiddenLayers[index - 1];

        return outputLayer;
    }
}