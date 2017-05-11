package domain;

import java.util.Arrays;
import java.util.Properties;
import java.util.concurrent.ThreadLocalRandom;

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

    public Properties getConfig() {
        return config;
    }

    public int getNumInputNeurons() {
        return numInputNeurons;
    }

    public int getNumHiddenNeurons() {
        return numHiddenNeurons;
    }

    public int getNumOutputNeurons() {
        return numOutputNeurons;
    }

    public float getIhBias() {
        return ihBias;
    }

    public float getHoBias() {
        return hoBias;
    }

    public float getLearningRate() {
        return learningRate;
    }

    public double getTermError() {
        return termError;
    }

    public Neuron[] getHiddenNeurons() {
        return hiddenNeurons;
    }

    public Neuron[] getOutputNeurons() {
        return outputNeurons;
    }

    public double[][] getWeightsIH() {
        return weightsIH;
    }

    public double[][] getWeightsHO() {
        return weightsHO;
    }

    public double[] getInputVector() {
        return inputVector;
    }

    public double[] getOutputVector() {
        return outputVector;
    }

    public double[] getTargetVector() {
        return targetVector;
    }

    public double getSampleError() {
        return sampleError;
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
        for (int i = 0; i < numInputNeurons; i++) {
            for (int j = 0; j < numHiddenNeurons; j++) {
                weightsIH[i][j] = ThreadLocalRandom.current().nextDouble(0.0d, 1.0d);
            }
        }

        for (int i = 0; i < numHiddenNeurons; i++) {
            for (int j = 0; j < numOutputNeurons; j++) {
                weightsHO[i][j] = ThreadLocalRandom.current().nextDouble(0.0d, 1.0d);
            }
        }
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

