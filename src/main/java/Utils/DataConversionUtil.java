package Utils;

/**
 * Created by sanketkorgaonkar on 5/11/17.
 */
public class DataConversionUtil {
    public static double[] convertDataVector(Double[] doubles) {
        double[] returnDoubles = new double[doubles.length];
        for (int i = 0; i < doubles.length; i++) {
            returnDoubles[i] = doubles[i].doubleValue();
        }
        return returnDoubles;
    }

    public static double[] roundOffValues(double[] doubles) {
        double[] returnDoubles = new double[doubles.length];
        for (int i = 0; i < doubles.length; i++) {
            returnDoubles[i] = Math.round(doubles[i]);
        }
        return returnDoubles;
    }

    public static boolean compareVectors(double[] vector1, double[] vector2) {
        if (vector1.length != vector2.length) {
            return false;
        }
        boolean equals = true;
        for (int i = 0; i < vector1.length; i++) {
            int a = (int) vector1[i];
            int b = (int) vector2[i];
            equals = (equals && (a == b));
        }
        return equals;
    }
}
