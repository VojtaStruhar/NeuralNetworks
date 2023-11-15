namespace NeuralNetworkProject;

public interface IActivationFunction
{
    public double Func(double x);

    public double Derivative(double x);
}

public static class Sigmoid
{
    public static double Func(double x) {
        return 1 / (1 + Math.Exp(-x));
    }

    public static double Derivative(double x) {
        return x * (1 - x);
    }
}

public static class ReLU
{
    // ReLU Function
    public static double Func(double x) {
        return Math.Max(0, x);
    }

    // Derivative of ReLU Function
    public static double Derivative(double x) {
        if (x > 0) return 1;
        return 0;
    }
}