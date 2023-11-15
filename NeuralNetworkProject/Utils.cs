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
        for (var i = 0; i < arr.Length; i++) {
            result[i] = new double[numberOfClasses];
            result[i][(int)arr[i]] = 1;
        }

        return result;
    }

    public static string FormatArray(double[] arr, int maxLength = 16) {
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
}