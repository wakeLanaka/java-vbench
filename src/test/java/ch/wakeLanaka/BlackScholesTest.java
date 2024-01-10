package ch.wakeLanaka;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import jdk.incubator.vector.GPUInformation;
import jdk.incubator.vector.SVMBuffer;
import java.util.Random;

public class BlackScholesTest {

    private static final GPUInformation SPECIES_SVM = SVMBuffer.SPECIES_PREFERRED;

    Random rand = new Random();

    float randFloat(float low, float high) {
        float val = rand.nextFloat();
        return (1.0f - val) * low + val * high;
    }

    float[] fillRandom(float low, float high, int size) {
        float[] array = new float[size];
        for (int i = 0; i < array.length; i++) {
            array[i] = randFloat(low, high);
        }
        return array;
    }


    @Test
    void BlackScholesvsBlackScholesAVX15() {
        final int SIZE = 15;

        float[] call = new float[SIZE];
        float[] put = new float[SIZE];
        float[] callAVX = new float[SIZE];
        float[] putAVX = new float[SIZE];

        float r = 0.02f;
        float sig = 0.30f;
        float[] s0 = fillRandom(5.0f, 30.0f, SIZE);
        float[] x = fillRandom(1.0f, 100.0f, SIZE);
        float[] t = fillRandom(0.25f, 10.0f, SIZE);

        BlackScholes.computeSerial(sig, r, x, call, put, t, s0, SIZE, 0);

        int offset = BlackScholes.computeAVX(sig, r, x, callAVX, putAVX, t, s0);
        if(offset < SIZE){
            BlackScholes.computeSerial(sig, r, x, callAVX, putAVX, t, s0, SIZE, offset);
        }

        for (var i = 0; i < SIZE; i++) {
            assertEquals(call[i], callAVX[i], 0.001f);
            assertEquals(put[i], putAVX[i], 0.001f);
        }
    }

    @Test
    void BlackScholesvsBlackScholesAVX16() {
        final int SIZE = 16;

        float[] call = new float[SIZE];
        float[] put = new float[SIZE];
        float[] callAVX = new float[SIZE];
        float[] putAVX = new float[SIZE];

        float r = 0.02f;
        float sig = 0.30f;
        float[] s0 = fillRandom(5.0f, 30.0f, SIZE);
        float[] x = fillRandom(1.0f, 100.0f, SIZE);
        float[] t = fillRandom(0.25f, 10.0f, SIZE);

        BlackScholes.computeSerial(sig, r, x, call, put, t, s0, SIZE, 0);
        int offset = BlackScholes.computeAVX(sig, r, x, callAVX, putAVX, t, s0);
        BlackScholes.computeSerial(sig, r, x, callAVX, putAVX, t, s0, SIZE, offset);

        for (var i = 0; i < SIZE; i++) {
            assertEquals(call[i], callAVX[i], 0.001f);
            assertEquals(put[i], putAVX[i], 0.001f);
        }
    }

    @Test
    void BlackScholesvsBlackScholesAVX17() {
        final int SIZE = 17;

        float[] call = new float[SIZE];
        float[] put = new float[SIZE];
        float[] callAVX = new float[SIZE];
        float[] putAVX = new float[SIZE];

        float r = 0.02f;
        float sig = 0.30f;
        float[] s0 = fillRandom(5.0f, 30.0f, SIZE);
        float[] x = fillRandom(1.0f, 100.0f, SIZE);
        float[] t = fillRandom(0.25f, 10.0f, SIZE);

        BlackScholes.computeSerial(sig, r, x, call, put, t, s0, SIZE, 0);
        int offset = BlackScholes.computeAVX(sig, r, x, callAVX, putAVX, t, s0);
        BlackScholes.computeSerial(sig, r, x, callAVX, putAVX, t, s0, SIZE, offset);

        for (var i = 0; i < SIZE; i++) {
            assertEquals(call[i], callAVX[i], 0.001f);
            assertEquals(put[i], putAVX[i], 0.001f);
        }
    }

    @Test
    void BlackScholesvsBlackScholesAVX511() {
        final int SIZE = 511;

        float[] call = new float[SIZE];
        float[] put = new float[SIZE];
        float[] callAVX = new float[SIZE];
        float[] putAVX = new float[SIZE];

        float r = 0.02f;
        float sig = 0.30f;
        float[] s0 = fillRandom(5.0f, 30.0f, SIZE);
        float[] x = fillRandom(1.0f, 100.0f, SIZE);
        float[] t = fillRandom(0.25f, 10.0f, SIZE);

        BlackScholes.computeSerial(sig, r, x, call, put, t, s0, SIZE, 0);
        int offset = BlackScholes.computeAVX(sig, r, x, callAVX, putAVX, t, s0);
        BlackScholes.computeSerial(sig, r, x, callAVX, putAVX, t, s0, SIZE, offset);

        for (var i = 0; i < SIZE; i++) {
            assertEquals(call[i], callAVX[i], 0.001f);
            assertEquals(put[i], putAVX[i], 0.001f);
        }
    }

    @Test
    void BlackScholesvsBlackScholesAVX512() {
        final int SIZE = 512;

        float[] call = new float[SIZE];
        float[] put = new float[SIZE];
        float[] callAVX = new float[SIZE];
        float[] putAVX = new float[SIZE];

        float r = 0.02f;
        float sig = 0.30f;
        float[] s0 = fillRandom(5.0f, 30.0f, SIZE);
        float[] x = fillRandom(1.0f, 100.0f, SIZE);
        float[] t = fillRandom(0.25f, 10.0f, SIZE);

        BlackScholes.computeSerial(sig, r, x, call, put, t, s0, SIZE, 0);
        int offset = BlackScholes.computeAVX(sig, r, x, callAVX, putAVX, t, s0);
        BlackScholes.computeSerial(sig, r, x, callAVX, putAVX, t, s0, SIZE, offset);

        for (var i = 0; i < SIZE; i++) {
            assertEquals(call[i], callAVX[i], 0.001f);
            assertEquals(put[i], putAVX[i], 0.001f);
        }
    }

    @Test
    void BlackScholesvsBlackScholesAVX513() {
        final int SIZE = 513;

        float[] call = new float[SIZE];
        float[] put = new float[SIZE];
        float[] callAVX = new float[SIZE];
        float[] putAVX = new float[SIZE];

        float r = 0.02f;
        float sig = 0.30f;
        float[] s0 = fillRandom(5.0f, 30.0f, SIZE);
        float[] x = fillRandom(1.0f, 100.0f, SIZE);
        float[] t = fillRandom(0.25f, 10.0f, SIZE);

        BlackScholes.computeSerial(sig, r, x, call, put, t, s0, SIZE, 0);
        int offset = BlackScholes.computeAVX(sig, r, x, callAVX, putAVX, t, s0);
        BlackScholes.computeSerial(sig, r, x, callAVX, putAVX, t, s0, SIZE, offset);

        for (var i = 0; i < SIZE; i++) {
            assertEquals(call[i], callAVX[i], 0.001f);
            assertEquals(put[i], putAVX[i], 0.001f);
        }
    }

    @Test
    void BlackScholesvsBlackScholesSVM15() {
        final int SIZE = 15;

        float[] call = new float[SIZE];
        float[] put = new float[SIZE];
        float[] callSVM = new float[SIZE];
        float[] putSVM = new float[SIZE];

        float r = 0.02f;
        float sig = 0.30f;
        float[] s0 = fillRandom(5.0f, 30.0f, SIZE);
        float[] x = fillRandom(1.0f, 100.0f, SIZE);
        float[] t = fillRandom(0.25f, 10.0f, SIZE);
        var s0Buf = SVMBuffer.fromArray(SPECIES_SVM, s0);
        var xBuf = SVMBuffer.fromArray(SPECIES_SVM, x);
        var tBuf = SVMBuffer.fromArray(SPECIES_SVM, t);
        var callBuf = SVMBuffer.fromArray(SPECIES_SVM, call);
        var putBuf = SVMBuffer.fromArray(SPECIES_SVM, put);

        BlackScholes.computeSerial(sig, r, x, call, put, t, s0, SIZE, 0);
        BlackScholes.computeSVM(sig, r, xBuf, callSVM, putSVM, tBuf, s0Buf);

        for (var i = 0; i < SIZE; i++) {
            assertEquals(call[i], callSVM[i], 0.001f);
            assertEquals(put[i], putSVM[i], 0.001f);
        }
    }

    @Test
    void BlackScholesvsBlackScholesSVM16() {
        final int SIZE = 16;

        float[] call = new float[SIZE];
        float[] put = new float[SIZE];
        float[] callSVM = new float[SIZE];
        float[] putSVM = new float[SIZE];

        float r = 0.02f;
        float sig = 0.30f;
        float[] s0 = fillRandom(5.0f, 30.0f, SIZE);
        float[] x = fillRandom(1.0f, 100.0f, SIZE);
        float[] t = fillRandom(0.25f, 10.0f, SIZE);
        var s0Buf = SVMBuffer.fromArray(SPECIES_SVM, s0);
        var xBuf = SVMBuffer.fromArray(SPECIES_SVM, x);
        var tBuf = SVMBuffer.fromArray(SPECIES_SVM, t);
        var callBuf = SVMBuffer.fromArray(SPECIES_SVM, call);
        var putBuf = SVMBuffer.fromArray(SPECIES_SVM, put);

        BlackScholes.computeSerial(sig, r, x, call, put, t, s0, SIZE, 0);
        BlackScholes.computeSVM(sig, r, xBuf, callSVM, putSVM, tBuf, s0Buf);

        for (var i = 0; i < SIZE; i++) {
            assertEquals(call[i], callSVM[i], 0.001f);
            assertEquals(put[i], putSVM[i], 0.001f);
        }
    }

    @Test
    void BlackScholesvsBlackScholesSVM17() {
        final int SIZE = 17;

        float[] call = new float[SIZE];
        float[] put = new float[SIZE];
        float[] callSVM = new float[SIZE];
        float[] putSVM = new float[SIZE];

        float r = 0.02f;
        float sig = 0.30f;
        float[] s0 = fillRandom(5.0f, 30.0f, SIZE);
        float[] x = fillRandom(1.0f, 100.0f, SIZE);
        float[] t = fillRandom(0.25f, 10.0f, SIZE);
        var s0Buf = SVMBuffer.fromArray(SPECIES_SVM, s0);
        var xBuf = SVMBuffer.fromArray(SPECIES_SVM, x);
        var tBuf = SVMBuffer.fromArray(SPECIES_SVM, t);
        var callBuf = SVMBuffer.fromArray(SPECIES_SVM, call);
        var putBuf = SVMBuffer.fromArray(SPECIES_SVM, put);

        BlackScholes.computeSerial(sig, r, x, call, put, t, s0, SIZE, 0);
        BlackScholes.computeSVM(sig, r, xBuf, callSVM, putSVM, tBuf, s0Buf);

        for (var i = 0; i < SIZE; i++) {
            assertEquals(call[i], callSVM[i], 0.001f);
            assertEquals(put[i], putSVM[i], 0.001f);
        }
    }

    @Test
    void BlackScholesvsBlackScholesSVM511() {
        final int SIZE = 511;

        float[] call = new float[SIZE];
        float[] put = new float[SIZE];
        float[] callSVM = new float[SIZE];
        float[] putSVM = new float[SIZE];

        float r = 0.02f;
        float sig = 0.30f;
        float[] s0 = fillRandom(5.0f, 30.0f, SIZE);
        float[] x = fillRandom(1.0f, 100.0f, SIZE);
        float[] t = fillRandom(0.25f, 10.0f, SIZE);
        var s0Buf = SVMBuffer.fromArray(SPECIES_SVM, s0);
        var xBuf = SVMBuffer.fromArray(SPECIES_SVM, x);
        var tBuf = SVMBuffer.fromArray(SPECIES_SVM, t);
        var callBuf = SVMBuffer.fromArray(SPECIES_SVM, call);
        var putBuf = SVMBuffer.fromArray(SPECIES_SVM, put);

        BlackScholes.computeSerial(sig, r, x, call, put, t, s0, SIZE, 0);
        BlackScholes.computeSVM(sig, r, xBuf, callSVM, putSVM, tBuf, s0Buf);

        for (var i = 0; i < SIZE; i++) {
            assertEquals(call[i], callSVM[i], 0.001f);
            assertEquals(put[i], putSVM[i], 0.001f);
        }
    }

    @Test
    void BlackScholesvsBlackScholesSVM512() {
        final int SIZE = 512;

        float[] call = new float[SIZE];
        float[] put = new float[SIZE];
        float[] callSVM = new float[SIZE];
        float[] putSVM = new float[SIZE];

        float r = 0.02f;
        float sig = 0.30f;
        float[] s0 = fillRandom(5.0f, 30.0f, SIZE);
        float[] x = fillRandom(1.0f, 100.0f, SIZE);
        float[] t = fillRandom(0.25f, 10.0f, SIZE);
        var s0Buf = SVMBuffer.fromArray(SPECIES_SVM, s0);
        var xBuf = SVMBuffer.fromArray(SPECIES_SVM, x);
        var tBuf = SVMBuffer.fromArray(SPECIES_SVM, t);
        var callBuf = SVMBuffer.fromArray(SPECIES_SVM, call);
        var putBuf = SVMBuffer.fromArray(SPECIES_SVM, put);

        BlackScholes.computeSerial(sig, r, x, call, put, t, s0, SIZE, 0);
        BlackScholes.computeSVM(sig, r, xBuf, callSVM, putSVM, tBuf, s0Buf);

        for (var i = 0; i < SIZE; i++) {
            assertEquals(call[i], callSVM[i], 0.001f);
            assertEquals(put[i], putSVM[i], 0.001f);
        }
    }

    @Test
    void BlackScholesvsBlackScholesSVM513() {
        final int SIZE = 513;

        float[] call = new float[SIZE];
        float[] put = new float[SIZE];
        float[] callSVM = new float[SIZE];
        float[] putSVM = new float[SIZE];

        float r = 0.02f;
        float sig = 0.30f;
        float[] s0 = fillRandom(5.0f, 30.0f, SIZE);
        float[] x = fillRandom(1.0f, 100.0f, SIZE);
        float[] t = fillRandom(0.25f, 10.0f, SIZE);
        var s0Buf = SVMBuffer.fromArray(SPECIES_SVM, s0);
        var xBuf = SVMBuffer.fromArray(SPECIES_SVM, x);
        var tBuf = SVMBuffer.fromArray(SPECIES_SVM, t);
        var callBuf = SVMBuffer.fromArray(SPECIES_SVM, call);
        var putBuf = SVMBuffer.fromArray(SPECIES_SVM, put);

        BlackScholes.computeSerial(sig, r, x, call, put, t, s0, SIZE, 0);
        BlackScholes.computeSVM(sig, r, xBuf, callSVM, putSVM, tBuf, s0Buf);

        for (var i = 0; i < SIZE; i++) {
            assertEquals(call[i], callSVM[i], 0.001f);
            assertEquals(put[i], putSVM[i], 0.001f);
        }
    }

    @Test
    void BlackScholesvsBlackScholesBuilder15() {
        final int SIZE = 15;

        float[] call = new float[SIZE];
        float[] put = new float[SIZE];
        float[] callBuilder = new float[SIZE];
        float[] putBuilder = new float[SIZE];

        float r = 0.02f;
        float sig = 0.30f;
        float[] s0 = fillRandom(5.0f, 30.0f, SIZE);
        float[] x = fillRandom(1.0f, 100.0f, SIZE);
        float[] t = fillRandom(0.25f, 10.0f, SIZE);

        BlackScholes.computeSerial(sig, r, x, call, put, t, s0, SIZE, 0);
        BlackScholes.computeKernelBuilder(sig, r, x, callBuilder, putBuilder, t, s0);

        for (var i = 0; i < SIZE; i++) {
            assertEquals(call[i], callBuilder[i], 0.001f);
            assertEquals(put[i], putBuilder[i], 0.001f);
        }
    }

    @Test
    void BlackScholesvsBlackScholesBuilder16() {
        final int SIZE = 16;

        float[] call = new float[SIZE];
        float[] put = new float[SIZE];
        float[] callBuilder = new float[SIZE];
        float[] putBuilder = new float[SIZE];

        float r = 0.02f;
        float sig = 0.30f;
        float[] s0 = fillRandom(5.0f, 30.0f, SIZE);
        float[] x = fillRandom(1.0f, 100.0f, SIZE);
        float[] t = fillRandom(0.25f, 10.0f, SIZE);

        BlackScholes.computeSerial(sig, r, x, call, put, t, s0, SIZE, 0);
        BlackScholes.computeKernelBuilder(sig, r, x, callBuilder, putBuilder, t, s0);

        for (var i = 0; i < SIZE; i++) {
            assertEquals(call[i], callBuilder[i], 0.001f);
            assertEquals(put[i], putBuilder[i], 0.001f);
        }
    }

    @Test
    void BlackScholesvsBlackScholesBuilder17() {
        final int SIZE = 17;

        float[] call = new float[SIZE];
        float[] put = new float[SIZE];
        float[] callBuilder = new float[SIZE];
        float[] putBuilder = new float[SIZE];

        float r = 0.02f;
        float sig = 0.30f;
        float[] s0 = fillRandom(5.0f, 30.0f, SIZE);
        float[] x = fillRandom(1.0f, 100.0f, SIZE);
        float[] t = fillRandom(0.25f, 10.0f, SIZE);

        BlackScholes.computeSerial(sig, r, x, call, put, t, s0, SIZE, 0);
        BlackScholes.computeKernelBuilder(sig, r, x, callBuilder, putBuilder, t, s0);

        for (var i = 0; i < SIZE; i++) {
            assertEquals(call[i], callBuilder[i], 0.001f);
            assertEquals(put[i], putBuilder[i], 0.001f);
        }
    }

    @Test
    void BlackScholesvsBlackScholesBuilder511() {
        final int SIZE = 511;

        float[] call = new float[SIZE];
        float[] put = new float[SIZE];
        float[] callBuilder = new float[SIZE];
        float[] putBuilder = new float[SIZE];

        float r = 0.02f;
        float sig = 0.30f;
        float[] s0 = fillRandom(5.0f, 30.0f, SIZE);
        float[] x = fillRandom(1.0f, 100.0f, SIZE);
        float[] t = fillRandom(0.25f, 10.0f, SIZE);

        BlackScholes.computeSerial(sig, r, x, call, put, t, s0, SIZE, 0);
        BlackScholes.computeKernelBuilder(sig, r, x, callBuilder, putBuilder, t, s0);

        for (var i = 0; i < SIZE; i++) {
            assertEquals(call[i], callBuilder[i], 0.001f);
            assertEquals(put[i], putBuilder[i], 0.001f);
        }
    }

    @Test
    void BlackScholesvsBlackScholesBuilder512() {
        final int SIZE = 512;

        float[] call = new float[SIZE];
        float[] put = new float[SIZE];
        float[] callBuilder = new float[SIZE];
        float[] putBuilder = new float[SIZE];

        float r = 0.02f;
        float sig = 0.30f;
        float[] s0 = fillRandom(5.0f, 30.0f, SIZE);
        float[] x = fillRandom(1.0f, 100.0f, SIZE);
        float[] t = fillRandom(0.25f, 10.0f, SIZE);

        BlackScholes.computeSerial(sig, r, x, call, put, t, s0, SIZE, 0);
        BlackScholes.computeKernelBuilder(sig, r, x, callBuilder, putBuilder, t, s0);

        for (var i = 0; i < SIZE; i++) {
            assertEquals(call[i], callBuilder[i], 0.001f);
            assertEquals(put[i], putBuilder[i], 0.001f);
        }
    }

    @Test
    void BlackScholesvsBlackScholesBuilder513() {
        final int SIZE = 513;

        float[] call = new float[SIZE];
        float[] put = new float[SIZE];
        float[] callBuilder = new float[SIZE];
        float[] putBuilder = new float[SIZE];

        float r = 0.02f;
        float sig = 0.30f;
        float[] s0 = fillRandom(5.0f, 30.0f, SIZE);
        float[] x = fillRandom(1.0f, 100.0f, SIZE);
        float[] t = fillRandom(0.25f, 10.0f, SIZE);

        BlackScholes.computeSerial(sig, r, x, call, put, t, s0, SIZE, 0);
        BlackScholes.computeKernelBuilder(sig, r, x, callBuilder, putBuilder, t, s0);

        for (var i = 0; i < SIZE; i++) {
            assertEquals(call[i], callBuilder[i], 0.001f);
            assertEquals(put[i], putBuilder[i], 0.001f);
        }
    }
}
