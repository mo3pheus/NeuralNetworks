package domain;

/**
 * Created by sanketkorgaonkar on 5/10/17.
 */
public interface CanLearn {
    public double computeActivation(double net);
    public double computeActivationDerivative(double net);
    public double computeOutput(double net);
    public double computeInput(double[][] weightMatrix);
}
