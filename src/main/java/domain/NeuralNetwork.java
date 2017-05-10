package domain;

import java.util.Properties;

/**
 * Created by sanketkorgaonkar on 5/10/17.
 */
public class NeuralNetwork {
    private Properties config           = null;
    private int        numInputNeurons  = 0;
    private int        numHiddenNeurons = 0;
    private int        numOutputNeurons = 0;
    private int        iterationNumber  = 0;
    private float      ihBias           = 0.0f;
    private float      hoBias           = 0.0f;
    private float      learningRate     = 0.0f;
    private double     termError        = 0.0d;
    private double[]   inputVector      = null;
    private double[]   outputVector     = null;
    private Neuron[]   hiddenNeurons    = null;
    private Neuron[]   outputNeurons    = null;
    private double[][] weightsIH        = null;
    private double[][] weightsHO        = null;


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
    }

}

