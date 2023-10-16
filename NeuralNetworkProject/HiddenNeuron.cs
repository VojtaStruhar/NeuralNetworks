namespace NeuralNetworkProject;

public class HiddenNeuron : INeuron
{
    private float[] _weights;
    private INeuron[] _inputNeurons;

    private float _threshold = 0f;

    public HiddenNeuron(INeuron[] layerBeneath) {
        _inputNeurons = layerBeneath;
        _weights = new float[layerBeneath.Count()];
    }


    private float InnerPotential() {
        var weightedSum = 0f;
        for (var i = 0; i < _inputNeurons.Length; i++) {
            var w_i = _weights[i];
            var x_i = _inputNeurons[i].GetOutput();

            weightedSum += w_i * x_i;
        }

        return weightedSum;
    }

    private float ActivationFunction(float arg) {
        if (arg >= _threshold) return 1f;
        return 0f;
    }


    public float GetOutput() {
        return ActivationFunction(InnerPotential());
    }
}