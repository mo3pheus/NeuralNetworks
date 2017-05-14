package domain;

import java.util.Arrays;
import java.util.Properties;
import java.util.concurrent.ThreadLocalRandom;

/**
 * Created by sanketkorgaonkar on 5/10/17.
 */
public class NeuralNetwork {
    /* Network variables */
    private int        numInputNeurons  = 0;
    private int        numHiddenNeurons = 0;
    private int        numOutputNeurons = 0;
    private double     ihBias           = 0.0d;
    private double     hoBias           = 0.0d;
    private double     learningRate     = 0.0d;
    private double     momentum         = 0.0d;
    private double     termError        = 0.0d;
    private double     cumulativeError  = Double.MAX_VALUE;
    private Properties config           = null;
    private Neuron[]   hiddenNeurons    = null;
    private Neuron[]   outputNeurons    = null;
    private double[][] weightsIH        = null;
    private double[][] weightsHO        = null;
    private double[][] deltaWeightsIH   = null;
    private double[][] deltaWeightsHO   = null;

    /* Sample variables */
    private double[] inputVector         = null;
    private double[] outputVector        = null;
    private double[] targetVector        = null;
    private double   sampleError         = Double.MAX_VALUE;
    private int      numSamplesProcessed = 0;

    public NeuralNetwork(Properties config) {
        this.config = config;

        this.numInputNeurons = Integer.parseInt(this.config.getProperty("neural.networks.number.input.neurons"));
        this.numHiddenNeurons = Integer.parseInt(this.config.getProperty("neural.networks.number.hidden.neurons"));
        this.numOutputNeurons = Integer.parseInt(this.config.getProperty("neural.networks.number.output.neurons"));
        this.ihBias = Double.parseDouble(this.config.getProperty("neural.networks.ih.bias"));
        this.hoBias = Double.parseDouble(this.config.getProperty("neural.networks.ho.bias"));
        this.learningRate = Double.parseDouble(this.config.getProperty("neural.networks.learning.rate"));
        this.termError = Double.parseDouble(this.config.getProperty("neural.networks.termination.error"));
        this.momentum = Double.parseDouble(this.config.getProperty("neural.networks.backProp.momentum"));

        hiddenNeurons = new Neuron[numHiddenNeurons];
        outputNeurons = new Neuron[numOutputNeurons];

        weightsIH = new double[numInputNeurons][numHiddenNeurons];
        weightsHO = new double[numHiddenNeurons][numOutputNeurons];
        deltaWeightsIH = new double[numInputNeurons][numHiddenNeurons];
        deltaWeightsHO = new double[numHiddenNeurons][numOutputNeurons];

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
        cumulativeError = (numSamplesProcessed == 0) ? sampleError : (cumulativeError + sampleError);
        numSamplesProcessed++;
    }

    public void processSample(double[] inpVector, double[] opVector) throws Exception {
        this.inputVector = inpVector;
        this.targetVector = opVector;

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
        cumulativeError = (numSamplesProcessed == 0) ? sampleError : (cumulativeError + sampleError);
        numSamplesProcessed++;
    }

    public void adjustWeights() {
        double[][] deltaWHO = computeDeltaWHO();
        double[][] deltaWIH = computeDeltaWIH();

        /* Adjust who weights */
        for (int i = 0; i < numHiddenNeurons; i++) {
            for (int j = 0; j < numOutputNeurons; j++) {
                weightsHO[i][j] -= (learningRate * deltaWHO[i][j]);
                weightsHO[i][j] += (momentum * deltaWeightsHO[i][j]);
                deltaWeightsHO[i][j] = (momentum * deltaWeightsHO[i][j]) - (learningRate * deltaWHO[i][j]);
            }
        }

        /* Adjust wih weights */
        for (int i = 0; i < numInputNeurons; i++) {
            for (int j = 0; j < numHiddenNeurons; j++) {
                weightsIH[i][j] -= (learningRate * deltaWIH[i][j]);
                weightsIH[i][j] += (momentum * deltaWeightsIH[i][j]);
                deltaWeightsIH[i][j] = (momentum * deltaWeightsIH[i][j]) - (learningRate * deltaWIH[i][j]);
            }
        }
    }

    public Properties getConfig() {
        return config;
    }

    public double getCumulativeError() {
        return cumulativeError;
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

    public double getIhBias() {
        return ihBias;
    }

    public double getHoBias() {
        return hoBias;
    }

    public double getLearningRate() {
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

    public void resetNetworkParams() {
        numSamplesProcessed = 0;
        cumulativeError = Double.MAX_VALUE;
        //sampleError = Double.MAX_VALUE;
    }

    private double[][] computeDeltaWHO() {
        double[][] deltaWHO = new double[numHiddenNeurons][numOutputNeurons];

        for (int i = 0; i < numHiddenNeurons; i++) {
            for (int j = 0; j < numOutputNeurons; j++) {
                deltaWHO[i][j] = (outputVector[j] - targetVector[j]) * outputNeurons[j].computeActivationDerivative
                        (outputNeurons[j].getNet())
                                 * hiddenNeurons[i].getOutput();
            }
        }

        return deltaWHO;
    }

    private double[][] computeDeltaWIH() {
        double[][] deltaWIH = new double[numInputNeurons][numHiddenNeurons];

        for (int i = 0; i < numInputNeurons; i++) {
            for (int j = 0; j < numHiddenNeurons; j++) {
                double d_Etotal_OHj = 0.0d;
                for (int k = 0; k < numOutputNeurons; k++) {
                    d_Etotal_OHj += ((outputNeurons[k].getOutput() - targetVector[k]) * outputNeurons[k]
                            .computeActivationDerivative(outputNeurons[k].getNet()) * weightsHO[j][k]);
                }
                deltaWIH[i][j] = d_Etotal_OHj * hiddenNeurons[j].computeActivationDerivative(hiddenNeurons[j].getNet
                        ()) * inputVector[i];
            }
        }

        return deltaWIH;
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
                weightsIH[i][j] = ThreadLocalRandom.current().nextDouble(-1.0d, 1.0d);
                deltaWeightsIH[i][j] = 0.0d;
            }
        }

        for (int i = 0; i < numHiddenNeurons; i++) {
            for (int j = 0; j < numOutputNeurons; j++) {
                weightsHO[i][j] = ThreadLocalRandom.current().nextDouble(-1.0d, 1.0d);
                deltaWeightsHO[i][j] = 0.0d;
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

