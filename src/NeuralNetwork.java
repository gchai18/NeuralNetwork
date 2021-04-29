import java.io.*;

/*
 * This module models a multi layer perceptron and calculates values of the output nodes of the network
 * given a specified number of nodes in each layer and values for the input nodes and the target values of the outputs.
 * Trains the neural network to minimize the error using backpropagation, which adjusts the values of the weights
 * by calculating the steepest descent of each of the weights.
 * Data is inputted and outputted with user-specified files.
 *
 * Methods in this file:
 * void setInput (double[] input)                                      Sets the input nodes to a given array of specified values
 * void setWeights (double[][][] weights)                              Sets the weights to a given array of specified values
 * double random (double low, double high)                             Generates a random double within a given range
 * void setRandomWeights (double low, double high)                     Sets the weights of the network to random values in a given range
 * void printWeights()                                                 Prints all the weights, separated by a line
 * void calculate()                                                    Calculates the values of all nodes in the network
 * double f (double x)                                                 Threshold function for calculating the value of the nodes
 * double fprime (double x)                                            Derivative of the threshold function
 * double error (int target)                                           Calculates the error for a single training set
 * double totalError()                                                 Calculates the total error for the complete training set
 * void printError()                                                   Prints the error for each training set, separated by a line
 * void printOutputs()                                                 Prints the error, target values, and output values for each training set
 * void backpropagation (int target, double lambda)                    Uses the backpropagation algorithm to change the weights
 * void train (double lambda, double lf, int epoch, double minError)   Trains the neural network to minimize the total error using backpropagation
 *
 * @author Gabriel Chai
 */
public class NeuralNetwork
{
    public int totalLayers;          // the number of layers in the neural network
    public double[][][] w;           // the values for the weights in the neural network
    public double[][] activation;    // the values of the nodes in the neural network
    public double[][] theta;         // the theta values used for backpropagation
    public double[][] psi;           // the psi values used for backpropagation
    public double[][] omega;         // the omega values used for backpropagation
    public double[][] inputValues;   // the values of the inputs nodes used to train the neural network
    public double[][] t;             // the target values for the output nodes for each of the input values
    public PrintWriter pw;           // the output file writer

    /*
     * Constructor for object of class NeuralNetwork.
     * Instantiates the arrays that store the values of the nodes and weights of the neural network.
     *
     * @param inputNodes the number of input nodes
     * @param hiddenLayerNodes the number of nodes in each hidden layer
     * @param outputNodes the number of output nodes
     */
    public NeuralNetwork(int inputNodes, int[] hiddenLayerNodes, int outputNodes, int inputSets, double[][] inputs, double[][] targets, PrintWriter pw)
    {
        int hiddenLayers = hiddenLayerNodes.length;
        totalLayers = hiddenLayers + 2;

        /*
         * Instantiates the jagged array that stores the values of the nodes
         * The first parameter is the layer and the second parameter is the specific node in the layer
         */
        activation = new double[totalLayers][];
        activation[0] = new double[inputNodes];
        for (int n = 0; n < hiddenLayers; n++)
        {
            activation[n + 1] = new double[hiddenLayerNodes[n]];
        }
        activation[totalLayers - 1] = new double[outputNodes];

        /*
         * Instantiates jagged arrays that store the theta, psi, and omega values used for backpropagation
         */
        theta = new double[totalLayers][];
        psi = new double[totalLayers][];
        omega = new double[totalLayers][];
        for (int n = 0; n < totalLayers; n++)
        {
            theta[n] = new double[activation[n].length];
            psi[n] = new double[activation[n].length];
            omega[n] = new double[activation[n].length];
        }

        /*
         * Instantiates the jagged array that stores the values of the weights
         * The first parameter is the initial layer, the second parameter is the node of the first layer,
         * and the third parameter is the node of the third layer
         */
        w = new double[totalLayers - 1][][];
        for (int n = 0; n < totalLayers - 1; n++)
        {
            w[n] = new double[activation[n].length][activation[n + 1].length];
        }

        // The input values for the neural network
        inputValues = new double[inputSets][inputNodes];
        for (int i = 0; i < inputSets; i++)
        {
            for (int j = 0; j < inputNodes; j++)
            {
                inputValues[i][j] = inputs[i][j];
            }
        }

        // the target values of the neural network
        t = new double[outputNodes][inputSets];
        for (int i = 0; i < outputNodes; i++)
        {
            for (int j = 0; j < inputSets; j++)
            {
                t[i][j] = targets[i][j];
            }
        }

        this.pw = pw;
    }

    /*
     * Sets the input node values to a given array of values
     *
     * @param input the input values
     */
    public void setInput(double[] input)
    {
        for (int k = 0; k < activation[0].length; k++)
        {
            activation[0][k] = input[k];
        }
    }

    /*
     * Sets the weights of the neural network to a given array of weights
     *
     * @param weights the values of the weights in the neural network
     */
    public void setWeights(double[][][] weights)
    {
        for (int n = 0; n < weights.length; n++)
        {
            for (int k = 0; k < weights[n].length; k++)
            {
                for (int j = 0; j < weights[n][k].length; j++)
                {
                    w[n][k][j] = weights[n][k][j];
                }
            }
        }
    }

    /*
     * Generates a random double within a given range
     *
     * @param low the lower bound of the random value
     * @param high the upper bound of the random value
     */
    public double random(double low, double high)
    {
        double diff = high - low;
        return Math.random() * diff + low;
    }

    /*
     * Sets the weights of the neural network to a random value within a given range
     *
     * @param low the lower bound of the random value
     * @param high the upper bound of the random value
     */
    public void setRandomWeights(double low, double high)
    {
        for (int n = 0; n < w.length; n++)
        {
            for (int k = 0; k < w[n].length; k++)
            {
                for (int j = 0; j < w[n][k].length; j++)
                {
                    w[n][k][j] = random(low, high);
                }
            }
        }
    }

    /*
     * Prints all values of the weights, separated by a line
     */
    public void printWeights()
    {
        for (int n = 0; n < w.length; n++)
        {
            for (int k = 0; k < w[n].length; k++)
            {
                for (int j = 0; j < w[n][k].length; j++)
                {
                    pw.println("w[" + n + "][" + k + "][" + j + "]: " + w[n][k][j] + " ");
                }
            }
        }
    }

    /*
     * Calculates the values of all the nodes in the neural network
     * The value of a node is the activation function of the dot product of all the nodes
     * in the previous layer and the weights going from the nodes.
     * The code that has been commented out is used to debug this method.
     */
    public void calculate()
    {
        for (int n = 1; n < activation.length; n++)
        {
            for (int k = 0; k < activation[n].length; k++)
            {
                theta[n][k] = 0.0;
                //pw.print("activation[" + n + "][" + k + "]: f(");
                for (int j = 0; j < activation[n - 1].length; j++)
                {
                    theta[n][k] += (activation[n - 1][j] * w[n - 1][j][k]);
                    //pw.print("activation[" + (n-1) + "][" + j + "]w[" + (n-1) + "][" + j + "][" + k + "] + ");
                }
                //pw.println();
                activation[n][k] = f(theta[n][k]);
            }
        }
    }


    /*
     * The threshold function for calculating the value of the nodes, which is the sigmoid function.
     *
     * @param x the input of the function
     * @return the resulting value of the function
     */
    public double f(double x)
    {
        return 1.0 / (1.0 + Math.exp(-x));
    }

    /*
     * The derivative of the threshold function for calculating the value of the nodes.
     *
     * @param x the input of the function
     * @return the output of the function
     */
    public double fprime(double x)
    {
        double num = f(x);
        return num * (1.0 - num);
    }

    /*
     * Calculates the error for a single training set
     *
     * @param target the index of the set of target values of the output nodes
     * @return the error of the neural network for a single training set
     */
    public double error(int target)
    {
        double error = 0.0;
        for (int i = 0; i < activation[totalLayers - 1].length; i++)
        {
            double diff = (t[i][target] - activation[totalLayers - 1][i]);
            error += (diff * diff);
        }
        error = error / 2.0;
        return error;
    }

    /*
     * Calculates the total error for the neural network for the complete training set
     *
     * @return the error of the neural network for the complete training set
     */
    public double totalError()
    {
        double sum = 0.0;
        for (int i = 0; i < inputValues.length; i++)
        {
            setInput(inputValues[i]);
            calculate();
            double error = error(i);
            sum += (error * error);
        }
        sum = Math.sqrt(sum);
        return sum;
    }

    /*
     * Prints the error for each training set, separated by a line
     */
    public void printError()
    {
        for (int i = 0; i < inputValues.length; i++)
        {
            setInput(inputValues[i]);
            calculate();
            pw.println("Error: " + error(i));
        }
    }

    /*
     * Prints the error, target values, and output values for each set of inputs
     */
    public void printOutputs()
    {
        for (int i = 0; i < inputValues.length; i++)
        {
            // Calculates the value of the outputs
            setInput(inputValues[i]);
            calculate();

            // Prints out the values of the input nodes
            pw.print("Inputs: ");
            for (int j = 0; j < inputValues[i].length; j++)
            {
                pw.print(inputValues[i][j] + " ");
            }
            pw.println();

            // Prints the target and output values and the error
            pw.println("Error: " + error(i));
            for (int j = 0; j < activation[totalLayers - 1].length; j++)
            {
                pw.println("Target: " + t[j][i] + " Output: " + activation[totalLayers - 1][j]);
            }

            pw.println();
        }
    }

    /*
     * Uses the backpropagation algorithm to calculate gradient of the error function with respect to the weights
     * and changes the weights in order to minimize the error for a specified training set.
     *
     * @param target the index for the set of target values for the outputs
     * @param lambda the learning factor that adjusts the changes in weights
     */
    public void backpropagation(int target, double lambda)
    {
        // calculating the changes for the right-most set of weights
        for (int i = 0; i < activation[totalLayers - 1].length; i++)
        {
            omega[totalLayers - 1][i] = t[i][target] - activation[totalLayers - 1][i];
            psi[totalLayers - 1][i] = omega[totalLayers - 1][i] * fprime(theta[totalLayers - 1][i]);

            // updating the weights of the network
            for (int j = 0; j < activation[totalLayers - 2].length; j++)
            {
                w[totalLayers - 2][j][i] += lambda * activation[totalLayers - 2][j] * psi[totalLayers - 1][i];
            }
        }

        // calculating the changes for the other weights
        for (int n = totalLayers - 2; n > 0; n--)
        {
            for (int j = 0; j < activation[n].length; j++)
            {
                omega[n][j] = 0.0;
                for (int i = 0; i < activation[n + 1].length; i++)
                {
                    omega[n][j] += (psi[n + 1][i] * w[n][j][i]);
                }

                psi[n][j] = omega[n][j] * fprime(theta[n][j]);

                // updating the weights of the network
                for (int k = 0; k < activation[n - 1].length; k++)
                {
                    w[n - 1][k][j] += lambda * activation[n - 1][k] * psi[n][j];
                }
            } // for (int j=0; j<activation[n].length; j++)
        } // for (int n=totalLayers-2; n>0; n--)
    }


    /*
     * Minimizes the total error of the neural network by calculating the gradient descent for each weight using backpropagation,
     * adjusting the weights, checking to see if the error decreases, modifies the learning factor, and does it again
     * until the total error is below a specified threshold
     *
     * @param lambda the learning factor is multiplied to the changes in weights
     * @param lf the amount that the learning factor changes
     * @param epoch the number of times that the network is trained
     * @param minError the minimum error threshold
     */
    public void train(double lambda, double lf, int epoch, double minError)
    {
        int times = 0; // the number of times the network has been trained
        while (totalError() > minError && times < epoch)
        {
            int target = times % t[0].length;

            // calculates the current error for a given set of input values
            setInput(inputValues[target]);
            calculate();
            double oldError = error(target);

            // changes the weights and calculates the new error
            backpropagation(target, lambda);
            calculate();
            double newError = error(target);

            // decreases lambda if the error increases; if the error decreases, lambda increases
            if (newError > oldError)
            {
                lambda /= lf;
            }
            else
            {
                lambda *= lf;
            }

            times++;
        } // while (totalError() > minError && times < epoch)

        // Prints the number of times run, lambda, total error, and the outputs
        pw.println("Times Run: " + times);
        pw.println("lambda = " + lambda);
        pw.println("Total Error = " + totalError());
        pw.println();
        printOutputs();

        // Prints termination condition
        pw.print("Reason for ending: ");
        if (times == epoch)
        {
            pw.println("Ran " + epoch + " times");
        }
        else
        {
            pw.println("Error is below minimum error threshold of " + minError);
        }
    }
}