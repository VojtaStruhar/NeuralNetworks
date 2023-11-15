using System.Text;

namespace NeuralNetworkProject;

public static class Utils
{
    public static void Shuffle<T>(Random rng, T[] array) {
        var n = array.Length;
        while (n > 1) {
            var k = rng.Next(n--);
            var temp = array[n];
            array[n] = array[k];
            array[k] = temp;
        }
    }

    public static double[][] OneHotEncode(double[] arr, int numberOfClasses) {
        var result = new double[arr.Length][];
        for (var i = 0; i < arr.Length; i++) result[i] = OneHotEncode(arr[i], numberOfClasses);

        return result;
    }

    public static double[] OneHotEncode(double element, int numberOfClasses) {
        var result = new double[numberOfClasses];
        result[(int)element] = 1;
        return result;
    }

    public static string FormatArray(double[] arr, int maxLength = 10) {
        var iterations = Math.Min(arr.Length, maxLength);
        var formattedOutput = new StringBuilder();
        formattedOutput.Append("[");
        for (var j = 0; j < iterations; j++) {
            formattedOutput.Append(Math.Round(arr[j], 4));
            if (j < iterations - 1)
                formattedOutput.Append(", ");
        }

        if (arr.Length > 16)
            formattedOutput.Append(", ...");

        formattedOutput.Append("]");
        return formattedOutput.ToString();
    }

    public static double[] Softmax(double[] values) {
        var expSum = 0.0;
        for (var i = 0; i < values.Length; i++) expSum += Math.Exp(values[i]);

        var result = new double[values.Length];
        for (var i = 0; i < values.Length; i++) result[i] = Math.Exp(values[i]) / expSum;

        return result;
    }
}