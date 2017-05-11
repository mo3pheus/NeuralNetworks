package domain;

import java.util.Arrays;
import java.util.Properties;

/**
 * Created by sanketkorgaonkar on 5/10/17.
 */
public class NeuralNetwork {
    /* Network variables */
    private Properties config           = null;
    private int        numInputNeurons  = 0;
    private int        numHiddenNeurons = 0;
    private int        numOutputNeurons = 0;
    private float      ihBias           = 0.0f;
    private float      hoBias           = 0.0f;
    private float      learningRate     = 0.0f;
    private double     termError        = 0.0d;
    private Neuron[]   hiddenNeurons    = null;
    private Neuron[]   outputNeurons    = null;
    private double[][] weightsIH        = null;
    private double[][] weightsHO        = null;

    /* Sample variables */
    private double[] inputVector  = null;
    private double[] outputVector = null;
    private double[] targetVector = null;
    private double   sampleError  = 0.0d;

    public NeuralNetwork(Properties config) {
        this.config = config;

        this.numInputNeurons = Integer.parseInt(this.config.getProperty("neural.networks.number.input.neurons"));
        this.numHiddenNeurons = Integer.parseInt(this.config.getProperty("neural.networks.number.hidden.neurons"));
        this.numOutputNeurons = Integer.parseInt(this.config.getProperty("neural.networks.number.output.neurons"));
        this.ihBias = Float.parseFloat(this.config.getProperty("neural.networks.ih.bias"));
        this.hoBias = Float.parseFloat(this.config.getProperty("neural.networks.ho.bias"));
        this.learningRate = Float.parseFloat(this.config.getProperty("neural.networks.learning.rate"));
        this.termError = Double.parseDouble(this.config.getProperty("neural.networks.termination.error"));

        hiddenNeurons = new Neuron[numHiddenNeurons];
        outputNeurons = new Neuron[numOutputNeurons];

        weightsIH = new double[numInputNeurons][numHiddenNeurons];
        weightsHO = new double[numHiddenNeurons][numOutputNeurons];

        inputVector = new double[numInputNeurons];
        outputVector = new double[numOutputNeurons];
        targetVector = new double[numOutputNeurons];

        initializeNeurons();
        initializeWeightMatrices();
    }

    public void processSample(String sample) throws Exception {
        String[] sampleElements = sample.split(",");

        if (sampleElements.length != (numInputNeurons + numOutputNeurons)) {
            throw new Exception("Sample is malformed - expecting length = " + numInputNeurons + numOutputNeurons);
        } else {
            parseSample(sampleElements);
        }

        double[] opHiddenNeurons = new double[numHiddenNeurons];
        Arrays.fill(opHiddenNeurons, 0.0d);

        /* Feed the hidden neurons */
        for (int i = 0; i < numHiddenNeurons; i++) {
            hiddenNeurons[i].setInputVector(inputVector);
            hiddenNeurons[i].computeInput(weightsIH);
            opHiddenNeurons[i] = hiddenNeurons[i].computeOutput(hiddenNeurons[i].getNet());
        }

        /* Feed the output neurons */
        for (int i = 0; i < numOutputNeurons; i++) {
            outputNeurons[i].setInputVector(opHiddenNeurons);
            outputNeurons[i].computeInput(weightsHO);
            outputVector[i] = outputNeurons[i].computeOutput(outputNeurons[i].getNet());
        }

        /* Compute the sample error */
        sampleError = NeuralNetwork.computeError(outputVector, targetVector);
    }

    private void initializeNeurons() {
        for (int i = 0; i < numHiddenNeurons; i++) {
            hiddenNeurons[i] = new Neuron(i);
            hiddenNeurons[i].setBias(ihBias);
        }

        for (int i = 0; i < numOutputNeurons; i++) {
            outputNeurons[i] = new Neuron(i);
            outputNeurons[i].setBias(hoBias);
        }
    }

    private void initializeWeightMatrices() {

    }

    private static double computeError(double[] outputVector, double[] targetVector) throws Exception {
        if (outputVector.length != targetVector.length) {
            throw new Exception("Target and output vectors don't match - go bury your head in the sand!");
        }

        int    length = outputVector.length;
        double error  = 0.0d;
        for (int i = 0; i < length; i++) {
            error += Math.pow((outputVector[i] - targetVector[i]), 2.0d);
        }
        error /= (double) length;

        return error;
    }

    private void parseSample(String[] sampleElements) {
        int sampleIndex = 0;
        for (int i = 0; i < inputVector.length; i++) {
            inputVector[i] = Double.parseDouble(sampleElements[sampleIndex++]);
        }

        for (int i = 0; i < targetVector.length; i++) {
            targetVector[i] = Double.parseDouble(sampleElements[sampleIndex++]);
        }
    }
}

