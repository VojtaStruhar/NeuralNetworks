namespace NeuralNetworkProject;

public static class CsvUtils
{
    public static double[][] ReadColors(string path) {
        var result = ReadVectors(path);
        for (var i = 0; i < result.Length; i++)
        for (var j = 0; j < result[i].Length; j++)
            result[i][j] /= 255.0; // Normalize the colors to [0, 1]
        return result;
    }

    public static double[][] ReadVectors(string path) {
        var lines = File.ReadAllLines(path);
        var result = new double[lines.Length][];

        for (var i = 0; i < lines.Length; i++) {
            var line = lines[i];
            var values = line.Split(',');
            result[i] = new double[values.Length];
            for (var j = 0; j < values.Length; j++)
                result[i][j] = double.Parse(values[j]);
        }

        return result;
    }

    public static double[] ReadLabels(string path) {
        var lines = File.ReadAllLines(path);
        var result = new double[lines.Length];

        for (var i = 0; i < lines.Length; i++) {
            var line = lines[i];
            result[i] = double.Parse(line);
        }

        return result;
    }
}