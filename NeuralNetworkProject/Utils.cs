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
}