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

    public static int[] blurAVX(int radius, float[] ys, BufferedImage image) {
        int width = image.getWidth();
        int height = image.getHeight();
        BufferedImage outputImage = new BufferedImage(width, height, image.getType());

        float sigma = (float)Math.max(radius / 2.0, 1.0);
        int kernelWidth = 2 * radius + 1;

        float[] kernel = new float[kernelWidth * kernelWidth];
        float exponentDenominator = 2 * sigma * sigma;
        float expressionDenominator = exponentDenominator * (float)Math.PI;
        var vExponentDenominator = FloatVector.broadcast(SPECIES, exponentDenominator);
        var vExpressionDenominator = FloatVector.broadcast(SPECIES, expressionDenominator);

        final int ysUpperBound = SPECIES.loopBound(ys.length);
        int i = 0;
        float sum = 0.0f;
        FloatVector vy;
        FloatVector vx;
        for (int x = -radius; x <= radius; x++) {
            vx = FloatVector.broadcast(SPECIES, x);
            int j = 0;
            for (; j < ysUpperBound; j += SPECIES.length()) {
                vy = FloatVector.fromArray(SPECIES, ys, j);
                var vExponentNumerator = vx.mul(vx).add(vy.mul(vy)).neg();
                var vEExpression = vExponentNumerator.div(vExponentDenominator).lanewise(VectorOperators.EXP);
                var vKernelValues = vEExpression.div(vExpressionDenominator);
                sum += vKernelValues.reduceLanes(VectorOperators.ADD);
                vKernelValues.intoArray(kernel, i);
                i+= SPECIES.length();
            }
            for(; j < ys.length; j++){
                float exponentNumerator = -(x * x + ys[j] * ys[j]);
                float eExpression = (float)Math.exp(exponentNumerator / exponentDenominator);
                float kernelValue = eExpression / expressionDenominator;
                kernel[i] = (float) kernelValue;
                sum += kernelValue;
                i++;
            }
        }

        final int kernelUpperBound = SPECIES.loopBound(kernel.length);
        int j = 0;
        FloatVector vKernel;
        for(; j < kernelUpperBound; j+= SPECIES.length()) {
            vKernel = FloatVector.fromArray(SPECIES, kernel, j);
            var normalized = vKernel.div(sum);
            normalized.intoArray(kernel, j);
        }
        for(; j < kernel.length; j++){
            kernel[j] /= sum;
        }

        int resultWidth = width - 2 * radius;
        int resultHeight = height - 2 * radius;
        int resultElements = resultWidth * resultHeight;
        int[] output = new int[resultElements];
        int counter = 0;
        for (int y = radius; y < height - radius; y++) {
            for (int x = radius; x < width - radius; x++) {
                var redValue = 0.0f;
                var greenValue = 0.0f;
                var blueValue = 0.0f;
                int[] imagePixels = new int[kernel.length];
                image.getRGB(x - radius, y - radius, kernelWidth, kernelWidth, imagePixels, 0, kernelWidth);

                for (j = 0; j < kernelUpperBound; j += SPECIES.length()) {
                    vKernel = FloatVector.fromArray(SPECIES, kernel, j);

                    for (int k = j; k < j + SPECIES.length(); k++) {
                        int pixel = imagePixels[k];
                        float red = (float) ((pixel >> 16) & 0xFF);
                        float green = (float) ((pixel >> 8) & 0xFF);
                        float blue = (float) (pixel & 0xFF);

                        redValue += red * kernel[k];
                        greenValue += green * kernel[k];
                        blueValue += blue * kernel[k];
                    }
                }

                for (j = kernelUpperBound; j < kernel.length; j++) {
                    int pixel = imagePixels[j];
                    float red = (float) ((pixel >> 16) & 0xFF);
                    float green = (float) ((pixel >> 8) & 0xFF);
                    float blue = (float) (pixel & 0xFF);

                    redValue += red * kernel[j];
                    greenValue += green * kernel[j];
                    blueValue += blue * kernel[j];
                }

                int newRGB = (clamp((int) redValue) << 16) | (clamp((int) greenValue) << 8) | clamp((int) blueValue);
                output[counter] = newRGB;
                counter++;
            }
        }
        return output;
    }

    public static int[] blurSVMFast(int radius, float[] ys, BufferedImage image){
        int width = image.getWidth();
        int height = image.getHeight();

        float sigma = (float)Math.max(radius / 2.0, 1.0);
        int kernelWidth = 2 * radius + 1;

        float exponentDenominator = 2 * sigma * sigma;
        float kernelDivision = exponentDenominator * (float)Math.PI;

        float sum = 0.0f;
        var vy = SVMBuffer.fromArray(SPECIES_SVM, ys);
        var vx = SVMBuffer.fromArray(SPECIES_SVM, ys);
        var vys = vy.mul(vy).repeatEachNumber(vy.length);
        var vxs = vx.repeatFullBuffer(vx.length).MultiplyRepeatInPlace(vx);
        var vKernelValues = vys.add(vxs).mulInPlace(-1);

        vKernelValues.divInPlace(exponentDenominator).expInPlace();
        vKernelValues.divInPlace(kernelDivision);

        sum += vKernelValues.sumReduceFloat();

        vKernelValues.divInPlace(sum);

        int[] imagePixels = new int[width * height];
        image.getRGB(0, 0, width, height, imagePixels, 0, width);

        var vImagePixelsInt = SVMBuffer.fromArray(SPECIES_SVM, imagePixels);

        var vbluesInt = vImagePixelsInt.and(0xFF);
        var vgreensInt = vImagePixelsInt.ashr(8).and(0xFF);
        var vredsInt = vImagePixelsInt.ashr(16).and(0xFF);

        var vbluesFloat = vbluesInt.toFloat();
        var vgreensFloat = vgreensInt.toFloat();
        var vredsFloat = vredsInt.toFloat();

        int resultWidth = width - 2 * radius;
        int resultHeight = height - 2 * radius;
        int resultelements = resultWidth * resultHeight;

        var newBluesFloat = vbluesFloat.eachAreaFMA(vKernelValues, width, kernelWidth, resultelements);
        var newGreensFloat = vgreensFloat.eachAreaFMA(vKernelValues, width, kernelWidth, resultelements);
        var newRedsFloat = vredsFloat.eachAreaFMA(vKernelValues, width, kernelWidth, resultelements);

        var clampedVB = newBluesFloat.max(0.0f).min(255.0f);
        var clampedVG = newGreensFloat.max(0.0f).min(255.0f);
        var clampedVR = newRedsFloat.max(0.0f).min(255.0f);

        var vbInt = clampedVB.toInt();
        var vgInt = clampedVG.toInt();
        var vrInt = clampedVR.toInt();

        var shiftedRed = vrInt.lshl(16);
        var shiftedGreen = vgInt.lshl(8);
        var colors = shiftedRed.or(shiftedGreen).orInPlace(vbInt);
        int[] outputPixels = new int[colors.length];
        colors.intoArray(outputPixels);
        BufferedImage outputImage = new BufferedImage(width, height, image.getType());
        outputImage.setRGB(radius, radius, resultWidth, resultHeight, outputPixels, 0, resultWidth);

        vy.releaseSVMBuffer();
        vx.releaseSVMBuffer();
        vys.releaseSVMBuffer();
        vxs.releaseSVMBuffer();
        clampedVB.releaseSVMBuffer();
        clampedVG.releaseSVMBuffer();
        clampedVR.releaseSVMBuffer();
        newBluesFloat.releaseSVMBuffer();
        newGreensFloat.releaseSVMBuffer();
        newRedsFloat.releaseSVMBuffer();
        vbluesFloat.releaseSVMBuffer();
        vgreensFloat.releaseSVMBuffer();
        vredsFloat.releaseSVMBuffer();
        vKernelValues.releaseSVMBuffer();
        colors.releaseSVMBuffer();
        shiftedRed.releaseSVMBuffer();
        shiftedGreen.releaseSVMBuffer();
        vbInt.releaseSVMBuffer();
        vgInt.releaseSVMBuffer();
        vrInt.releaseSVMBuffer();
        vbluesInt.releaseSVMBuffer();
        vgreensInt.releaseSVMBuffer();
        vredsInt.releaseSVMBuffer();
        vImagePixelsInt.releaseSVMBuffer();

        return outputPixels;
    }

    public static int[] blurSerial(int radius, BufferedImage image) {
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

        int resultWidth = width - 2 * radius;
        int resultHeight = height - 2 * radius;
        int resultElements = resultWidth * resultHeight;
        int[] output = new int[resultElements];
        var counter = 0;
        for (int y = radius; y < height - radius; y++) {
            for (int x = radius; x < width - radius; x++) {
                float redValue = 0.0f;
                float greenValue = 0.0f;
                float blueValue = 0.0f;

                int[] imagePixels = new int[kernel.length];
                image.getRGB(x - radius, y - radius, kernelWidth, kernelWidth, imagePixels, 0, kernelWidth);

                for(int i = 0; i < kernel.length; i++){
                    int pixel = imagePixels[i];
                    int blue =  pixel & 0xFF;
                    int green =  (pixel >> 8) & 0xFF;
                    int red =  (pixel >> 16) & 0xFF;
                    blueValue += blue * kernel[i];
                    greenValue += green * kernel[i];
                    redValue += red * kernel[i];
                }

                int newRGB = (clamp((int) redValue) << 16) | (clamp((int) greenValue) << 8) | clamp((int) blueValue);
                output[counter] = newRGB;
                counter++;
            }
        }

        return output;
    }

    private static int clamp(int value) {
        return Math.min(Math.max(value, 0), 255);
    }
}
