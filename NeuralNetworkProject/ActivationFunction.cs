namespace NeuralNetworkProject;

public abstract class ActivationFunction
{
    public abstract double Func(double x);

    public abstract double Derivative(double x);

    public static readonly ActivationFunction Sigmoid = new Sigmoid();
    public static readonly ActivationFunction ReLU = new ReLU();
    public static readonly ActivationFunction Invalid = new Invalid();
}

public class Sigmoid : ActivationFunction
{
    public override double Func(double x) {
        return 1 / (1 + Math.Exp(-x));
    }

    public override double Derivative(double x) {
        return x * (1 - x);
    }
}

public class ReLU : ActivationFunction
{
    // ReLU Function
    public override double Func(double x) {
        return Math.Max(0, x);
    }

    // Derivative of ReLU Function
    public override double Derivative(double x) {
        if (x > 0) return 1;
        return 0;
    }
}

public class Invalid : ActivationFunction
{
    public override double Func(double x) {
        throw new Exception("Invalid activation function");
    }

    public override double Derivative(double x) {
        throw new Exception("Invalid activation function derivative");
    }
}