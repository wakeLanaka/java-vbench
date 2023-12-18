package ch.wakeLanaka;

import jdk.incubator.vector.VectorOperators;
import java.awt.image.BufferedImage;
import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.IntVector;
import jdk.incubator.vector.VectorSpecies;
import jdk.incubator.vector.GPUInformation;
import jdk.incubator.vector.SVMBuffer;

public class GaussianBlur {

    private static VectorSpecies<Float> SPECIES = FloatVector.SPECIES_PREFERRED;
    private static GPUInformation SPECIES_SVM = SVMBuffer.SPECIES_PREFERRED;

    private static void printArray(float[] array, String info){
        for(int i = 0; i < array.length; i++){
            System.out.println(info + ": " + array[i]);
        }
    }

    private static void printArray(int[] array, String info){
        for(int i = 0; i < 50; i++){
            System.out.println(info + ": " + array[i]);
        }
    }

    public static BufferedImage blurAVX(int radius, float[] ys, BufferedImage image) {
        int width = image.getWidth();
        int height = image.getHeight();
        BufferedImage outputImage = new BufferedImage(width, height, image.getType());

        float sigma = (float)Math.max(radius / 2.0, 1.0);
        int kernelWidth = 2 * radius + 1;

        float[] kernel = new float[kernelWidth * kernelWidth];
        float exponentDenominator = 2 * sigma * sigma;
        var vExponentDenominator = FloatVector.broadcast(SPECIES, exponentDenominator);
        var vKernelDivision = FloatVector.broadcast(SPECIES, exponentDenominator * (float)Math.PI);

        final int ysUpperBound = SPECIES.loopBound(ys.length);
        int i = 0;
        float sum = 0.0f;
        for (int x = -radius; x <= radius; x++) {
            FloatVector vx = FloatVector.broadcast(SPECIES, x);
            int j = 0;
            for (; j < ysUpperBound; j += SPECIES.length()) {
                FloatVector vy = FloatVector.fromArray(SPECIES, ys, j);
                var vExponentNumerator = vx.mul(vx).add(vy.mul(vy)).neg();
                var vEExpression = vExponentNumerator.div(vExponentDenominator).lanewise(VectorOperators.EXP);
                var vKernelValues = vEExpression.div(vKernelDivision);
                sum += vKernelValues.reduceLanes(VectorOperators.ADD);
                vKernelValues.intoArray(kernel, i);
                i+= SPECIES.length();
            }
            for(; j < ys.length; j++){
                float exponentNumerator = -(x * x + ys[j] * ys[j]);
                float eExpression = (float)Math.exp(exponentNumerator / exponentDenominator);
                float kernelValue = eExpression / (2 * (float)Math.PI * sigma * sigma);
                kernel[i] = (float) kernelValue;
                sum += kernelValue;
                i++;
            }
        }

        final int kernelUpperBound = SPECIES.loopBound(kernel.length);
        int j = 0;
        for(; j < kernelUpperBound; j+= SPECIES.length()) {
            var vkernel = FloatVector.fromArray(SPECIES, kernel, j);
            var normalized = vkernel.div(sum);
            normalized.intoArray(kernel, j);
        }
        for(; j < kernel.length; j++){
            kernel[j] /= sum;
        }

        for (int x = radius; x < width - radius; x++) {
            for (int y = radius; y < height - radius; y++) {
                var redValue = 0.0f;
                var greenValue = 0.0f;
                var blueValue = 0.0f;
                int[] imagePixels = new int[kernel.length];
                float[] reds = new float[kernel.length];
                float[] greens = new float[kernel.length];
                float[] blues = new float[kernel.length];
                image.getRGB(x - radius, y - radius, kernelWidth, kernelWidth, imagePixels, 0, kernelWidth);
                getColors(imagePixels, reds, greens, blues);

                for(j = 0; j < kernelUpperBound; j+= SPECIES.length()) {
                    var vKernel = FloatVector.fromArray(SPECIES, kernel, j);
                    var vreds = FloatVector.fromArray(SPECIES, reds, j);
                    var vgreens = FloatVector.fromArray(SPECIES, greens, j);
                    var vblues = FloatVector.fromArray(SPECIES, blues, j);
                    redValue += vreds.mul(vKernel).reduceLanes(VectorOperators.ADD);
                    greenValue += vgreens.mul(vKernel).reduceLanes(VectorOperators.ADD);
                    blueValue += vblues.mul(vKernel).reduceLanes(VectorOperators.ADD);
                }
                for(; j < kernel.length; j++){
                    redValue += reds[j] * kernel[j];
                    greenValue += greens[j] * kernel[j];
                    blueValue += blues[j] * kernel[j];
                }
                int newRGB = (clamp((int) redValue) << 16) | (clamp((int) greenValue) << 8) | clamp((int) blueValue);
                outputImage.setRGB(x, y, newRGB);
            }
        }
        return outputImage;
    }

    private static void getColors(int[] imageArray, float[] reds, float[] greens, float[] blues) {
        for(int x = 0; x < imageArray.length; x++){
            for (int i = 0; i < 3; i++) {
                if(i == 0){
                    reds[x] = (float)((imageArray[x] >> 16) & 0xFF);
                } else if(i == 1){
                    greens[x] = (float)((imageArray[x] >> 8) & 0xFF);
                } else {
                    blues[x] = (float)(imageArray[x] & 0xFF);
                }
            }
        }
    }

    public static BufferedImage blurSVM(int radius, float[] ys, float[] xs, BufferedImage image){
        int width = image.getWidth();
        int height = image.getHeight();
        BufferedImage outputImage = new BufferedImage(width, height, image.getType());

        float sigma = (float)Math.max(radius / 2.0, 1.0);
        int kernelWidth = 2 * radius + 1;

        float exponentDenominator = 2 * sigma * sigma;
        float kernelDivision = exponentDenominator * (float)Math.PI;

        float sum = 0.0f;
        var vy = SVMBuffer.fromArray(SPECIES_SVM, ys);
        var vx = SVMBuffer.fromArray(SPECIES_SVM, xs);
        var vys = vy.mul(vy).repeat2(vy.length);
        var vxs = vx.repeat1(vx.length).MultiplyInPlaceRepeat(vx);
        var vKernelValues = vys.add(vxs).mulInPlace(-1);
        vy.releaseSVMBuffer();
        vx.releaseSVMBuffer();
        vys.releaseSVMBuffer();
        vxs.releaseSVMBuffer();
        vKernelValues.divInPlace(exponentDenominator).expInPlace();
        vKernelValues.divInPlace(kernelDivision);

        sum += vKernelValues.sumReduce();

        var normalized = vKernelValues.divInPlace(sum);

        float[] test = new float[normalized.length];
        var vreds = SVMBuffer.fromArray(SPECIES_SVM, test);
        var vgreens = SVMBuffer.fromArray(SPECIES_SVM, test);
        var vblues = SVMBuffer.fromArray(SPECIES_SVM, test);

        for (int x = radius; x < width - radius; x++) {
            for (int y = radius; y < height - radius; y++) {
                int[] imagePixels = new int[normalized.length];
                float[] reds = new float[normalized.length];
                float[] greens = new float[normalized.length];
                float[] blues = new float[normalized.length];
                image.getRGB(x - radius, y - radius, kernelWidth, kernelWidth, imagePixels, 0, kernelWidth);
                getColors(imagePixels, reds, greens, blues);

                vreds.reuse(reds);
                vgreens.reuse(greens);
                vblues.reuse(blues);

                var redValue = vreds.mulInPlace(normalized).sumReduce();
                var greenValue = vgreens.mulInPlace(normalized).sumReduce();
                var blueValue = vblues.mulInPlace(normalized).sumReduce();

                int newRGB = (clamp((int) redValue) << 16) | (clamp((int) greenValue) << 8) | clamp((int) blueValue);
                outputImage.setRGB(x, y, newRGB);
            }
        }
        vreds.releaseSVMBuffer();
        vgreens.releaseSVMBuffer();
        vblues.releaseSVMBuffer();

        return outputImage;
    }

    // public static BufferedImage blurSerial(int radius, BufferedImage image) {
    //     int width = image.getWidth();
    //     int height = image.getHeight();
    //     BufferedImage outputImage = new BufferedImage(width, height, image.getType());

    //     float sigma = (float)Math.max(radius / 2.0, 1.0);
    //     int kernelWidth = 2 * radius + 1;

    //     float[] kernel = new float[kernelWidth * kernelWidth];
    //     float sum = 0.0f;

    //     for (int x = -radius, i = 0; x <= radius; x++) {
    //         for (int y = -radius; y <= radius; y++, i++) {
    //             float exponentNumerator = -(x * x + y * y);
    //             float exponentDenominator = 2 * sigma * sigma;

    //             float eExpression = (float)Math.exp(exponentNumerator / exponentDenominator);
    //             float kernelValue = eExpression / (2 * (float)Math.PI * sigma * sigma);

    //             kernel[i] = (float) kernelValue;
    //             sum += kernelValue;
    //         }
    //     }

    //     for (int i = 0; i < kernel.length; i++) {
    //         kernel[i] /= (float) sum;
    //     }

    //     // printArray(kernel, "Serial");

    //     for (int y = radius; y < /* height -  */radius + 1; y++) {
    //         for (int x = radius; x < /* width -  */radius + 1; x++) {

    //             float redValue = 0.0f;
    //             float greenValue = 0.0f;
    //             float blueValue = 0.0f;

    //             for (int kernelY = -radius, i = 0; kernelY <= radius; kernelY++) {
    //                 for (int kernelX = -radius; kernelX <= radius; kernelX++, i++) {

    //                     float kernelValue = kernel[i];

    //                     int pixelRGB = image.getRGB(x + kernelX, y + kernelY);
    //                     int red = (pixelRGB >> 16) & 0xFF;
    //                     int green = (pixelRGB >> 8) & 0xFF;
    //                     int blue = pixelRGB & 0xFF;

    //                     redValue += red * kernelValue;
    //                     greenValue += green * kernelValue;
    //                     blueValue += blue * kernelValue;
    //                 }
    //             }

    //             System.out.println("redValue: " + redValue);
    //             System.out.println("greenValue: " + greenValue);
    //             System.out.println("blueValue: " + blueValue);

    //             int newRGB = (clamp((int) redValue) << 16) | (clamp((int) greenValue) << 8) | clamp((int) blueValue);
    //             outputImage.setRGB(x, y, newRGB);
    //         }
    //     }

    //     return outputImage;
    // }

    public static BufferedImage blurSerialWorking(int radius, BufferedImage image) {
        int width = image.getWidth();
        int height = image.getHeight();
        BufferedImage outputImage = new BufferedImage(width, height, image.getType());

        float sigma = (float)Math.max(radius / 2.0, 1.0);
        int kernelWidth = 2 * radius + 1;

        float[] kernel = new float[kernelWidth * kernelWidth];
        float sum = 0.0f;

        for (int x = -radius, i = 0; x <= radius; x++) {
            for (int y = -radius; y <= radius; y++, i++) {
                float exponentNumerator = -(x * x + y * y);
                float exponentDenominator = 2 * sigma * sigma;

                float eExpression = (float)Math.exp(exponentNumerator / exponentDenominator);
                float kernelValue = eExpression / (2 * (float)Math.PI * sigma * sigma);

                kernel[i] = (float) kernelValue;
                sum += kernelValue;
            }
        }

        for (int i = 0; i < kernel.length; i++) {
            kernel[i] /= (float) sum;
        }

        // printArray(kernel, "Serial");

        for (int x = radius; x < width - radius; x++) {
            for (int y = radius; y < height - radius; y++) {

                float redValue = 0.0f;
                float greenValue = 0.0f;
                float blueValue = 0.0f;

                for (int kernelX = -radius, i = 0; kernelX <= radius; kernelX++) {
                    for (int kernelY = -radius; kernelY <= radius; kernelY++, i++) {

                        float kernelValue = kernel[i];

                        int pixelRGB = image.getRGB(x + kernelX, y + kernelY);
                        int red = (pixelRGB >> 16) & 0xFF;
                        int green = (pixelRGB >> 8) & 0xFF;
                        int blue = pixelRGB & 0xFF;

                        redValue += red * kernelValue;
                        greenValue += green * kernelValue;
                        blueValue += blue * kernelValue;
                    }
                }
                // System.out.println(redValue);

                int newRGB = (clamp((int) redValue) << 16) | (clamp((int) greenValue) << 8) | clamp((int) blueValue);
                outputImage.setRGB(x, y, newRGB);
            }
        }

        return outputImage;
    }

    private static int clamp(int value) {
        return Math.min(Math.max(value, 0), 255);
    }
}
