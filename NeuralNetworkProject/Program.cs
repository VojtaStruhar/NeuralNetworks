// See https://aka.ms/new-console-template for more information

using NeuralNetworkProject;

Console.WriteLine("Hello, World!");

var hidden = new int[] { 5, 5 };
var nn = new Network(3, hidden, 1);

Console.WriteLine(nn);