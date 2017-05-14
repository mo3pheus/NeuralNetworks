package driver;

import domain.NeuralNetwork;

import java.util.Arrays;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.IOException;
import java.net.URL;
import java.util.Properties;

/**
 * Created by sanket on 5/10/17.
 */
public class NNTest {
    public static void main(String[] args) {
        System.out.println("Welcome to Neural Networks!");
        Properties    neuralNetworkConfig = NNTest.getNeuralNetworkConfig();
        NeuralNetwork ann                 = new NeuralNetwork(neuralNetworkConfig);
        System.out.println(" WIH ::");
        System.out.println(Arrays.toString(ann.getWeightsIH()[0]));
        System.out.println(Arrays.toString(ann.getWeightsIH()[1]));

        System.out.println(" WHO ::");
        System.out.println(Arrays.toString(ann.getWeightsHO()[0]));
        System.out.println(Arrays.toString(ann.getWeightsHO()[1]));

        String[] exOrProblem = {"0.0d,0.0d,0.0d", "0.0d,1.0d,1.0d", "1.0d,0.0d,1.0d", "1.0d,1.0d,0.0d"};

        boolean cont = true;
        while (cont || ann.getSampleError() > ann.getTermError()) {
            cont = false;
            for (String sample : exOrProblem) {
                try {
                    ann.processSample(sample);
                    ann.adjustWeights();
                    //System.out.println("targetVector = " + Arrays.toString(ann.getTargetVector()));
                    // System.out.println("outputVector = " + Arrays.toString(ann.getOutputVector()));
                } catch (Exception e) {
                    e.printStackTrace();
                }
                System.out.println("========================================================");
                System.out.println("Cumulative Error = " + ann.getCumulativeError());
                ann.resetNetworkParams();
            }
        }
    }

    private static Properties getNeuralNetworkConfig() {
        try {
            URL             fileUrl         = NNTest.class.getResource("/networkConfig.properties");
            FileInputStream fileInputStream = new FileInputStream(fileUrl.getPath());
            Properties      networkConfig   = new Properties();
            networkConfig.load(fileInputStream);
            return networkConfig;
        } catch (FileNotFoundException fileNotFoundException) {
            fileNotFoundException.printStackTrace();
        } catch (IOException e) {
            e.printStackTrace();
        }
        return null;
    }
}
