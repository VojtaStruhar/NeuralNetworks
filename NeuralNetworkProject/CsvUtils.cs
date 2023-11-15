namespace NeuralNetworkProject;

public static class CsvUtils
{
    public static double[][] ReadColors(string path, int maxCount) {
        var result = new double[maxCount][];
        using (var reader = new StreamReader(path)) {
            string line;
            var i = 0;
            while ((line = reader.ReadLine()) != null) {
                if (i == maxCount) {
                    Console.WriteLine("Ending before the end of the file - read " + i + " labels.");
                    break;
                }

                var values = line.Split(',');
                var vector = new double[values.Length];
                for (var j = 0; j < values.Length; j++)
                    vector[j] = double.Parse(values[j]) / 255.0; // Normalize the grayscale colors

                result[i] = vector;
                i++;
            }

            if (i < maxCount) Console.WriteLine("Array underfilled - " + i + "/" + maxCount + " labels.");
        }

        return result;
    }

    public static double[][] ReadVectors(string path, int numberOfVectors) {
        var result = new double[numberOfVectors][];
        using (var reader = new StreamReader(path)) {
            string line;
            var i = 0;
            while ((line = reader.ReadLine()) != null) {
                var values = line.Split(',');
                var vector = new double[values.Length];
                for (var j = 0; j < values.Length; j++)
                    vector[j] = double.Parse(values[j]);

                result[i] = vector;
                i++;
            }
        }

        return result;
    }


    public static double[][] ReadAndEncodeLabels(string path, int numberOfClasses, int maxCount) {
        var result = new double[maxCount][];
        using (var reader = new StreamReader(path)) {
            string line;
            var i = 0;
            while ((line = reader.ReadLine()) != null) {
                if (i == maxCount) {
                    Console.WriteLine("Ending before the end of the file - read " + i + " labels.");
                    break;
                }

                result[i] = Utils.OneHotEncode(double.Parse(line), numberOfClasses);
                i++;
            }

            if (i < maxCount) Console.WriteLine("Array underfilled - " + i + "/" + maxCount + " labels.");
        }

        return result;
    }

    public static double[] ReadLabels(string path, int maxCount) {
        var result = new double[maxCount];
        using (var reader = new StreamReader(path)) {
            string line;
            var i = 0;
            while ((line = reader.ReadLine()) != null) {
                if (i == maxCount) {
                    Console.WriteLine("Ending before the end of the file - read " + i + " labels.");
                    break;
                }

                result[i] = double.Parse(line);
                i++;
            }

            if (i < maxCount) Console.WriteLine("Array underfilled - " + i + "/" + maxCount + " labels.");
        }

        return result;
    }

    public static void Write(double[][] data, string path) {
        using (var file = new StreamWriter(path)) {
            foreach (var row in data) {
                var line = string.Join(",", row);
                file.WriteLine(line);
            }
        }
    }
}