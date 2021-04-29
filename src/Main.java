import java.util.*;
import java.io.*;

/*
 * The Main class creates NeuralNetwork using data from a user-specified input file, trains the network, and
 * outputs the results to a user-specified output file.
 *
 * @author Gabriel Chai
 */
public class Main
{
    /*
     * Asks the user for the names of the file that specifies the input values of the network and the information used to train the network
     * Reads the values in the file and creates a model of neural network using the specified values calculates the outputs of the network.
     * Trains the network using backpropagation to minimize the error and outputs the resulting data to a file specified by the user.
     */
    public static void main(String[] args) throws IOException
    {
        // Asks the user for the names of the input and output files
        Scanner s = new Scanner(System.in);
        System.out.print("What is the name of the input file? ");
        String inputFile = s.next();
        System.out.print("What is the name of the output file? ");
        String outputFile = s.next();

        // How many nodes are in each layer
        Scanner sc = new Scanner(new File(inputFile));
        int hidden = sc.nextInt();
        int input = sc.nextInt();
        int[] hiddenNodes = new int[hidden];
        for (int i = 0; i < hidden; i++)
        {
            hiddenNodes[i] = sc.nextInt();
        }
        int output = sc.nextInt();

        // The minimum and maximum values of the randomized weights
        double low = sc.nextDouble();
        double high = sc.nextDouble();

        // The learning factor, the amount the learning factor changes, the maximum number of times run,
        // and the minimum error threshold for training the neural network
        double lambda = sc.nextDouble();
        double lf = sc.nextDouble();
        int epoch = sc.nextInt();
        double minError = sc.nextDouble();

        // The number of sets of inputs values of the neural network
        int inputSets = sc.nextInt();

        // The input values for the neural network
        double[][] inputValues = new double[inputSets][input];
        for (int i = 0; i < inputSets; i++)
        {
            for (int j = 0; j < input; j++)
            {
                inputValues[i][j] = sc.nextDouble();
            }
        }

        // the target values of the neural network
        double[][] targets = new double[output][inputSets];
        for (int i = 0; i < output; i++)
        {
            for (int j = 0; j < inputSets; j++)
            {
                targets[i][j] = sc.nextDouble();
            }
        }

        // the output file writer
        PrintWriter pw = new PrintWriter(new BufferedWriter(new FileWriter(outputFile)));

        NeuralNetwork network = new NeuralNetwork(input, hiddenNodes, output, inputSets, inputValues, targets, pw);
        network.setRandomWeights(low, high);
        network.train(lambda, lf, epoch, minError);

        pw.close();
    }
}