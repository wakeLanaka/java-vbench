package ch.wakeLanaka;

import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.nio.file.Paths;
import javax.imageio.ImageIO;
import org.openjdk.jmh.annotations.*;
import static ch.wakeLanaka.GeneratorHelpers.createRadiusValues;

public class GaussianBlurBenchmark {

    @State(Scope.Thread)
    public static class BenchmarkSetup {

        @Param({"8", "10", "12", "14"})
        public int radius;
        public BufferedImage input;
        public float[] ys;
        public int[] outputPixels;

        @Setup(Level.Trial)
        public void doSetup() throws Exception {
            var absolutePath = new File(".").getAbsolutePath();
            var path = Paths.get(absolutePath + "/src/test/java/ch/wakeLanaka/img/GaussianTest.png");
            File file = path.toFile().getAbsoluteFile();

            if (file.exists()){
                input = ImageIO.read(file);
            } else {
                throw new IOException("File (" + path + ") does not exist!");
            }

            ys = createRadiusValues(radius);
        }

        @TearDown(Level.Trial)
        public void doTearDown() throws Exception {
            int width = input.getWidth();
            int height = input.getHeight();
            int resultWidth = width - 2 * radius;
            int resultHeight = height - 2 * radius;
            int resultelements = resultWidth * resultHeight;
            BufferedImage outputImage = new BufferedImage(width, height, input.getType());
            outputImage.setRGB(radius, radius, resultWidth, resultHeight, outputPixels, 0, resultWidth);
            var absolutePath = new File(".").getAbsolutePath();
            var outputPath = Paths.get(absolutePath + "/src/jmh/java/ch/wakeLanaka/img/benchmarkBlurred.png");
            File outputfile = new File(outputPath.toString());
            ImageIO.write(outputImage, "png", outputfile);
            System.out.println("Image created! at " + outputPath);
        }
    }

    @Benchmark
    @BenchmarkMode(Mode.AverageTime)
    public void gaussianBlurSerial(BenchmarkSetup state) {
        state.outputPixels = GaussianBlur.blurSerial(state.radius, state.input);
    }

    @Benchmark
    @BenchmarkMode(Mode.AverageTime)
    public void gaussianBlurSVMFast(BenchmarkSetup state) throws Exception {
        state.outputPixels = GaussianBlur.blurSVMFast(state.radius, state.ys, state.input);
    }

    // @Benchmark
    // @BenchmarkMode(Mode.AverageTime)
    // public void gaussianBlurSVM(BenchmarkSetup state) throws Exception {
    //     state.outputPixels = GaussianBlur.blurSVM(state.radius, state.ys, state.input);
    // }

    @Benchmark
    @BenchmarkMode(Mode.AverageTime)
    public void gaussianBlurAVX(BenchmarkSetup state) throws Exception {
        state.outputPixels = GaussianBlur.blurAVX(state.radius, state.ys, state.input);
    }
}
