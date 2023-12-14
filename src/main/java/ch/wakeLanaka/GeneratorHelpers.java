package ch.wakeLanaka;

import java.util.Random;
import java.util.concurrent.ThreadLocalRandom;

public class GeneratorHelpers {

    public static float[] initFloatArray(int length){
        var floatArray = new float[length];

        Random rand = new Random();
        for(var i = 0; i<length; i++){
            floatArray[i] = rand.nextFloat();
        }
        return floatArray;
    }

    public static int[] initIntArray(int length){
        var intArray = new int[length];

        Random rand = new Random();
        for(var i = 0; i<length; i++){
            intArray[i] = rand.nextInt();
        }

        return intArray;
    }

    public static float[] newFloatRowMajorMatrix(int size) {
        Random rand = new Random();
        float[] matrix = new float[size];
        for (int i = 0; i < matrix.length; ++i) {
            matrix[i] = rand.nextFloat();
        }
        return matrix;
    }

    public static float[] iotaFloatRowMajorMatrix(int size) {
        Random rand = new Random();
        float[] matrix = new float[size];
        for (int i = 0; i < matrix.length; i++) {
            matrix[i] = i;
        }
        return matrix;
    }

    public static float[] newFloatColumnMajorMatrix(int rows, int columns) {
        Random rand = new Random();
        float[] matrix = new float[rows * columns];
        int counter = 1;
        for (int row = 0; row < rows; row++) {
            for (int col = 0; col < columns; col++) {
                int index = col * rows + row;
                matrix[index] = rand.nextFloat();
                counter++;
            }
        }
        return matrix;
    }

    public static float[] iotaFloatColumnMajorMatrix(int rows, int columns) {
        Random rand = new Random();
        float[] matrix = new float[rows * columns];
        int counter = 0;
        for (int row = 0; row < rows; row++) {
            for (int col = 0; col < columns; col++) {
                int index = col * rows + row;
                matrix[index] = counter;
                counter++;
            }
        }
        return matrix;
    }
}
