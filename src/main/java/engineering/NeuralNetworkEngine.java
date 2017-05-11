package engineering;

import Utils.DataConversionUtil;
import domain.NeuralNetwork;
import egen.solutions.ennate.egen.solutions.sml.driver.SanketML;
import ennate.egen.solutions.sml.domain.Data;

import java.io.File;
import java.io.IOException;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;
import java.util.Properties;

/**
 * Created by sanketkorgaonkar on 5/11/17.
 */
public class NeuralNetworkEngine {
    private static NeuralNetwork         neuralNetwork = null;
    private static Map<String, double[]> classMap      = null;

    public static void main(String[] args) {
        SanketML irisProblem = new SanketML();
        try {
            irisProblem.loadData(new File(NeuralNetworkEngine.class.getResource("/testInputFiles/iris.data.txt")
                    .getPath()), ",", 4);
            irisProblem.populateTrainTestSets(80);
            System.out.println(irisProblem.getTrainingData().size());
        } catch (IOException e) {
            e.printStackTrace();
        }
        Map<String, Integer> uniqueClasses = NeuralNetworkEngine.findUniqueClassIds(irisProblem);
        NeuralNetworkEngine.buildclassMap(uniqueClasses);

        neuralNetwork = new NeuralNetwork(setUpNetworkProperties(4, classMap.keySet().size()));

        /* Training phase */
        int numCycles = 0;
        for(Data sample:irisProblem.getTrainingData()) {
            while ((neuralNetwork.getSampleError() > neuralNetwork.getTermError()) && (numCycles < 1000000)) {
                try {
                    neuralNetwork.processSample(DataConversionUtil.convertDataVector(sample.getFields()), classMap
                            .get(sample.getClassId()));
                    neuralNetwork.adjustWeights();
                } catch (Exception e) {
                    System.out.println("Error while processing sample = " + sample.toString());
                }

                System.out.println("Sample error = " + neuralNetwork.getSampleError());
                //neuralNetwork.resetNetworkParams();
                numCycles++;
            }
            neuralNetwork.resetNetworkParams();
            System.out.println("===================================================================");
        }
       System.out.println("Neural Network has converged! Iterations taken = " + numCycles);

        /* Training phase */
       /* numCycles = 0;
        while ((neuralNetwork.getCumulativeError() > neuralNetwork.getTermError()) && (numCycles < 1000000)) {
            for (Data sample : irisProblem.getTrainingData()) {
                try {
                    neuralNetwork.processSample(DataConversionUtil.convertDataVector(sample.getFields()), classMap
                            .get(sample.getClassId()));
                    neuralNetwork.adjustWeights();
                } catch (Exception e) {
                    System.out.println("Error while processing sample = " + sample.toString());
                }
            }
            System.out.println("Cumulative error = " + neuralNetwork.getCumulativeError());
            neuralNetwork.resetNetworkParams();
            numCycles++;
        }
        System.out.println("Neural Network has converged! Iterations taken = " + numCycles);*/

        /* Testing phase */
        int accuratePredictions = 0;
        int numSamples          = 0;
        for (Data testSample : irisProblem.getTrainingData()) {
            try {
                neuralNetwork.processSample(DataConversionUtil.convertDataVector(testSample.getFields()),
                        classMap.get(testSample.getClassId()));
                System.out.println("===================================================================");
                System.out.println("Target Vector = " + Arrays.toString(classMap.get(testSample.getClassId())));
                System.out.println("Actual output = " + Arrays.toString(neuralNetwork
                        .getOutputVector()));
                System.out.println("Actual Rounded Output = " + Arrays.toString(DataConversionUtil.roundOffValues
                        (neuralNetwork
                                .getOutputVector())));
                accuratePredictions = (DataConversionUtil.compareVectors(classMap.get(testSample.getClassId()),
                        DataConversionUtil.roundOffValues
                                (neuralNetwork
                                        .getOutputVector()))) ? (accuratePredictions + 1) : accuratePredictions;
            } catch (Exception e) {
                e.printStackTrace();
            }
            numSamples++;
        }
        System.out.println("Accurate Predictions = " + accuratePredictions + " Total test samples = " + numSamples);
    }

    public static void buildclassMap(Map<String, Integer> uniqueClasses) {
        int numClasses = uniqueClasses.keySet().size();
        classMap = new HashMap<String, double[]>();
        double[] targetVector = new double[numClasses];
        int      i            = 0;
        for (String classId : uniqueClasses.keySet()) {
            targetVector[i++] = 1.0d;
            classMap.put(classId, targetVector);
            targetVector = new double[numClasses];
        }
    }

    private static Map<String, Integer> findUniqueClassIds(SanketML problem) {
        Map<String, Integer> classes = new HashMap<String, Integer>();
        for (Data data : problem.getTrainingData()) {
            classes.put(data.getClassId(), new Integer(1));
        }
        return classes;
    }

    private static Properties setUpNetworkProperties(int numInputs, int numOutputs) {
        Properties networkConfig = new Properties();
        networkConfig.put("neural.networks.number.input.neurons", Integer.toString(numInputs));
        networkConfig.put("neural.networks.number.output.neurons", Integer.toString(numOutputs));
        networkConfig.put("neural.networks.number.hidden.neurons", Integer.toString(numOutputs));
        networkConfig.put("neural.networks.learning.rate", Double.toString(0.5d));
        networkConfig.put("neural.networks.ih.bias", Double.toString(1.8d));
        networkConfig.put("neural.networks.ho.bias", Double.toString(1.8d));
        networkConfig.put("neural.networks.termination.error", Double.toString(0.0001d));

        return networkConfig;
    }
}
