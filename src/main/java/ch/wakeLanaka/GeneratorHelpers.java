package ch.wakeLanaka;

public class GeneratorHelpers {

    private static float value = 0.25f;

    public static float[] initFloatArray(int length){
        var floatArray = new float[length];

        for(var i = 0; i<length; i++){
            floatArray[i] = value;
        }
        return floatArray;
    }

    public static float[] newFloatRowMajorMatrix(int size) {
        float[] matrix = new float[size];
        for (int i = 0; i < matrix.length; ++i) {
            matrix[i] = value + i;
        }
        return matrix;
    }

    public static float[] newFloatColumnMajorMatrix(int rows, int columns) {
        float[] matrix = new float[rows * columns];
        int counter = 0;
        for (int row = 0; row < rows; row++) {
            for (int col = 0; col < columns; col++) {
                int index = col * rows + row;
                matrix[index] = value + counter;
                counter++;
            }
        }
        return matrix;
    }

    public static float[] iotaFloatArray(int size){
        var array = new float[size];
        for(int i = 0; i < size; i++){
            array[i] = i * 1.0f;
        }
        return array;
    }
}
