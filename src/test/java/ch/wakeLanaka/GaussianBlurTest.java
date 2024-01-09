package ch.wakeLanaka;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import jdk.incubator.vector.GPUInformation;
import jdk.incubator.vector.SVMBuffer;
import java.awt.image.BufferedImage;
import java.io.File;
import java.io.IOException;
import java.nio.file.Paths;
import javax.imageio.ImageIO;

public class GaussianBlurTest {

    private static final GPUInformation SPECIES_SVM = SVMBuffer.SPECIES_PREFERRED;

    private static int width;
    private static int height;

    public BufferedImage createImage() throws Exception {
        var absolutePath = new File(".").getAbsolutePath();
        var path = Paths.get(absolutePath + "/src/test/java/ch/wakeLanaka/img/GaussianTest.png");
        File file = path.toFile().getAbsoluteFile();
        BufferedImage image;
        if (file.exists()) {
            image = ImageIO.read(file);
        } else {
            throw new IOException("File (" + path + ") does not exist!");
        }
        width = image.getWidth();
        height = image.getHeight();
        return image;
    }

    @Test
    void GaussianBlurvsGaussianBlurAVX7() throws Exception {
        final int radius = 7;

        var input = createImage();

        float[] ys = GeneratorHelpers.createRadiusValues(radius);

        var outputSerial = GaussianBlur.blurSerial(radius, input);
        var outputAVX = GaussianBlur.blurAVX(radius, ys, input);

        for (var i = 0; i < outputSerial.length; i++) {
            assertEquals(outputSerial[i], outputAVX[i]);
        }
    }

    @Test
    void GaussianBlurvsGaussianBlurAVX10() throws Exception {
        final int radius = 10;

        var input = createImage();

        float[] ys = GeneratorHelpers.createRadiusValues(radius);

        var outputSerial = GaussianBlur.blurSerial(radius, input);
        var outputAVX = GaussianBlur.blurAVX(radius, ys, input);

        for (var i = 0; i < outputSerial.length; i++) {
            assertEquals(outputSerial[i], outputAVX[i]);
        }
    }

    @Test
    void GaussianBlurvsGaussianBlurSVMFast7() throws Exception {
        final int radius = 7;

        var input = createImage();

        float[] ys = GeneratorHelpers.createRadiusValues(radius);

        var outputSerial = GaussianBlur.blurSerial(radius, input);
        var outputSVM = GaussianBlur.blurSVMFast(radius, ys, input);

        for (var i = 0; i < outputSerial.length; i++) {
            assertEquals(outputSerial[i], outputSVM[i]);
        }
    }

    @Test
    void GaussianBlurvsGaussianBlurSVMFast10() throws Exception {
        final int radius = 10;

        var input = createImage();

        float[] ys = GeneratorHelpers.createRadiusValues(radius);

        var outputSerial = GaussianBlur.blurSerial(radius, input);
        var outputSVM = GaussianBlur.blurSVMFast(radius, ys, input);

        for (var i = 0; i < outputSerial.length; i++) {
            assertEquals(outputSerial[i], outputSVM[i]);
        }
    }
}
