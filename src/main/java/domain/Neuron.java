package domain;

/**
 * Created by sanketkorgaonkar on 5/10/17.
 */
public class Neuron implements CanLearn {
    private int      index;
    private double[] inputVector;
    private double   net;
    private double   bias;
    private double   output;

    public Neuron(int index) {
        this.index = index;
    }

    public double[] getInputVector() {
        return inputVector;
    }

    public void setInputVector(double[] inputVector) {
        this.inputVector = inputVector;
    }

    public double getBias() {
        return bias;
    }

    public void setBias(double bias) {
        this.bias = bias;
    }

    public double computeActivation(double net) {
        return 1.0d / (1.0d - Math.exp(-1.0d * net));
    }

    public double computeActivationDerivative(double net) {
        return computeActivation(net) * (1.0d - computeActivation(net));
    }

    public double computeOutput(double net) {
        output = computeActivation(net);
        return output;
    }

    public double computeInput(double[][] weightMatrix) {
        double net = 0.0d;
        for (int i = 0; i < weightMatrix[0].length; i++) {
            net += weightMatrix[index][i] * inputVector[i];
        }
        net += bias;

        this.net = net;

        return net;
    }

    public double getNet() {
        return net;
    }

    public double getOutput() {
        return output;
    }
}
