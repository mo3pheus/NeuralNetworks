package domain;

/**
 * Created by sanketkorgaonkar on 5/10/17.
 */
public class Neuron implements CanLearn {
    private int      index;
    private double[] inputs;
    private double   bias;

    public Neuron(int index) {
        this.index = index;
    }

    public double[] getInputs() {
        return inputs;
    }

    public void setInputs(double[] inputs) {
        this.inputs = inputs;
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
        return computeActivation(net);
    }

    public double computeInput(double[][] weightMatrix) {
        double net = 0.0d;
        for (int i = 0; i < weightMatrix[0].length; i++) {
            net += weightMatrix[index][i] * inputs[i];
        }
        net += bias;

        return net;
    }
}
