package ch.wakeLanaka;

import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.nio.file.Paths;
import javax.imageio.ImageIO;
import org.openjdk.jmh.annotations.*;


public class GaussianBlurBenchmark {

    @State(Scope.Thread)
    public static class BenchmarkSetup {

        @Param({"5"})
        public int radius;
        public BufferedImage input;
        public BufferedImage output;
        public float[] ys;

        @Setup(Level.Trial)
        public void doSetup() throws Exception {
            var path = Paths.get("/home/reto/Pictures/Wallpapers/japan_store.png");
            File file = path.toFile().getAbsoluteFile();
            if (file.exists()){
                input = ImageIO.read(file);
            } else {
                throw new IOException("File (" + path + ") does not exist!");
            }

            ys = new float[radius * 2 + 1];
            int x = 0;
            for(int i = -radius; i <= radius; i++) {
                ys[x] = (float)i;
                x++;
            }
        }

        @TearDown(Level.Trial)
        public void doTearDown() throws Exception {
            System.out.println("1");
            File outputfile = new File("/home/reto/Pictures/blured.png");
            System.out.println("2");
            ImageIO.write(output, "png", outputfile);
        }
    }

    // @Benchmark
    // @BenchmarkMode(Mode.AverageTime)
    // public void gaussianBlurSerial(BenchmarkSetup state) {
    //     state.output = GaussianBlur.blurSerial(state.radius, state.input);
    // }

    // @Benchmark
    // @BenchmarkMode(Mode.AverageTime)
    // public void gaussianBlurSerial(BenchmarkSetup state) throws Exception {
    //     state.output = GaussianBlur.blurSerialWorking(state.radius, state.input);
    //     // File outputfile = new File("/home/reto/Pictures/blured.png");
    //     // ImageIO.write(state.output, "png", outputfile);
    //     // System.out.println("Created!");
    // }

    @Benchmark
    @BenchmarkMode(Mode.AverageTime)
    public void gaussianBlurSVM(BenchmarkSetup state) throws Exception {
        state.output = GaussianBlur.blurSVM(state.radius, state.ys, state.ys, state.input);
        File outputfile = new File("/home/reto/Pictures/blured.png");
        ImageIO.write(state.output, "png", outputfile);
        System.out.println("Created!");
    }

    // @Benchmark
    // @BenchmarkMode(Mode.AverageTime)
    // public void TEST(BenchmarkSetup state) throws Exception {
    //     state.output = GaussianBlur.blurAVX(state.radius, state.ys, state.input);
    //     // System.out.println("---SERIAL---");
    //     // state.output = GaussianBlur.blurSerial(state.radius, state.input);
    //     // File outputfile = new File("/home/reto/Pictures/blured.png");
    //     // ImageIO.write(state.output, "png", outputfile);
    //     // System.out.println("Created!");
    // }
}
