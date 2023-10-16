namespace NeuralNetworkProject;

public class InputNeuron : INeuron
{
    private float _value;

    public InputNeuron(float value) {
        _value = value;
    }

    public float GetOutput() {
        return _value;
    }
}