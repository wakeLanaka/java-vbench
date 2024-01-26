package ch.wakeLanaka;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import jdk.incubator.vector.GPUInformation;
import jdk.incubator.vector.SVMBuffer;

public class FMATest {

    private static final GPUInformation SPECIES_SVM = SVMBuffer.SPECIES_PREFERRED;

    @Test
    void FMAvsFMAAVX15() {
        final int SIZE = 15;

        var a = GeneratorHelpers.initFloatArray(SIZE);

        var result = FmaArray.scalarFMA(a, a);
        var avx = FmaArray.vectorFMA(a, a);

        for (var i = 0; i < SIZE; i++) {
            assertEquals(result[i], avx[i], 0.001f);
        }
    }

    @Test
    void FMAvsFMAAVX16() {
        final int SIZE = 16;

        var a = GeneratorHelpers.initFloatArray(SIZE);

        var result = FmaArray.scalarFMA(a, a);
        var avx = FmaArray.vectorFMA(a, a);

        for (var i = 0; i < SIZE; i++) {
            assertEquals(result[i], avx[i], 0.001f);
        }
    }

    @Test
    void FMAvsFMAAVX17() {
        final int SIZE = 17;

        var a = GeneratorHelpers.initFloatArray(SIZE);

        var result = FmaArray.scalarFMA(a, a);
        var avx = FmaArray.vectorFMA(a, a);

        for (var i = 0; i < SIZE; i++) {
            assertEquals(result[i], avx[i], 0.001f);
        }
    }

    @Test
    void FMAvsFMAAVX511() {
        final int SIZE = 511;

        var a = GeneratorHelpers.initFloatArray(SIZE);

        var result = FmaArray.scalarFMA(a, a);
        var avx = FmaArray.vectorFMA(a, a);

        for (var i = 0; i < SIZE; i++) {
            assertEquals(result[i], avx[i], 0.001f);
        }
    }

    @Test
    void FMAvsFMAAVX512() {
        final int SIZE = 512;

        var a = GeneratorHelpers.initFloatArray(SIZE);

        var result = FmaArray.scalarFMA(a, a);
        var avx = FmaArray.vectorFMA(a, a);

        for (var i = 0; i < SIZE; i++) {
            assertEquals(result[i], avx[i], 0.001f);
        }
    }

    @Test
    void FMAvsFMAAVX513() {
        final int SIZE = 513;

        var a = GeneratorHelpers.initFloatArray(SIZE);

        var result = FmaArray.scalarFMA(a, a);
        var avx = FmaArray.vectorFMA(a, a);

        for (var i = 0; i < SIZE; i++) {
            assertEquals(result[i], avx[i], 0.001f);
        }
    }

    @Test
    void FMAvsFMAGPUCopy15() {
        final int SIZE = 15;

        var a = GeneratorHelpers.initFloatArray(SIZE);

        var result = FmaArray.scalarFMA(a, a);
        var gpu = FmaArray.gpuFMACopy(a, a);

        for (var i = 0; i < SIZE; i++) {
            assertEquals(result[i], gpu[i], 0.001f);
        }
    }

    @Test
    void FMAvsFMAGPUCopy16() {
        final int SIZE = 16;

        var a = GeneratorHelpers.initFloatArray(SIZE);

        var result = FmaArray.scalarFMA(a, a);
        var gpu = FmaArray.gpuFMACopy(a, a);

        for (var i = 0; i < SIZE; i++) {
            assertEquals(result[i], gpu[i], 0.001f);
        }
    }

    @Test
    void FMAvsFMAGPUCopy17() {
        final int SIZE = 17;

        var a = GeneratorHelpers.initFloatArray(SIZE);

        var result = FmaArray.scalarFMA(a, a);
        var gpu = FmaArray.gpuFMACopy(a, a);

        for (var i = 0; i < SIZE; i++) {
            assertEquals(result[i], gpu[i], 0.001f);
        }
    }

    @Test
    void FMAvsFMAGPUCopy511() {
        final int SIZE = 511;

        var a = GeneratorHelpers.initFloatArray(SIZE);

        var result = FmaArray.scalarFMA(a, a);
        var gpu = FmaArray.gpuFMACopy(a, a);

        for (var i = 0; i < SIZE; i++) {
            assertEquals(result[i], gpu[i], 0.001f);
        }
    }

    @Test
    void FMAvsFMAGPUCopy512() {
        final int SIZE = 512;

        var a = GeneratorHelpers.initFloatArray(SIZE);

        var result = FmaArray.scalarFMA(a, a);
        var gpu = FmaArray.gpuFMACopy(a, a);

        for (var i = 0; i < SIZE; i++) {
            assertEquals(result[i], gpu[i], 0.001f);
        }
    }

    @Test
    void FMAvsFMAGPUCopy513() {
        final int SIZE = 513;

        var a = GeneratorHelpers.initFloatArray(SIZE);

        var result = FmaArray.scalarFMA(a, a);
        var gpu = FmaArray.gpuFMACopy(a, a);

        for (var i = 0; i < SIZE; i++) {
            assertEquals(result[i], gpu[i], 0.001f);
        }
    }

    @Test
    void FMAvsFMAGPU15() {
        final int SIZE = 15;

        var a = GeneratorHelpers.initFloatArray(SIZE);
        var resGPU = new float[SIZE];
        var bufferA = SVMBuffer.fromArray(SPECIES_SVM, a);
        var bufferC = SVMBuffer.zero(SPECIES_SVM, bufferA.length, bufferA.type);

        var result = FmaArray.scalarFMA(a, a);
        var gpu = FmaArray.gpuFMA(bufferA, bufferA, bufferC);
        gpu.intoArray(resGPU);

        for (var i = 0; i < SIZE; i++) {
            assertEquals(result[i], resGPU[i], 0.001f);
        }
    }

    @Test
    void FMAvsFMAGPU16() {
        final int SIZE = 16;

        var a = GeneratorHelpers.initFloatArray(SIZE);
        var resGPU = new float[SIZE];
        var bufferA = SVMBuffer.fromArray(SPECIES_SVM, a);
        var bufferC = SVMBuffer.zero(SPECIES_SVM, bufferA.length, bufferA.type);

        var result = FmaArray.scalarFMA(a, a);
        var gpu = FmaArray.gpuFMA(bufferA, bufferA, bufferC);
        gpu.intoArray(resGPU);

        for (var i = 0; i < SIZE; i++) {
            assertEquals(result[i], resGPU[i], 0.001f);
        }
    }

    @Test
    void FMAvsFMAGPU17() {
        final int SIZE = 17;

        var a = GeneratorHelpers.initFloatArray(SIZE);
        var resGPU = new float[SIZE];
        var bufferA = SVMBuffer.fromArray(SPECIES_SVM, a);
        var bufferC = SVMBuffer.zero(SPECIES_SVM, bufferA.length, bufferA.type);

        var result = FmaArray.scalarFMA(a, a);
        var gpu = FmaArray.gpuFMA(bufferA, bufferA, bufferC);
        gpu.intoArray(resGPU);

        for (var i = 0; i < SIZE; i++) {
            assertEquals(result[i], resGPU[i], 0.001f);
        }
    }

    @Test
    void FMAvsFMAGPU511() {
        final int SIZE = 511;

        var a = GeneratorHelpers.initFloatArray(SIZE);
        var resGPU = new float[SIZE];
        var bufferA = SVMBuffer.fromArray(SPECIES_SVM, a);
        var bufferC = SVMBuffer.zero(SPECIES_SVM, bufferA.length, bufferA.type);

        var result = FmaArray.scalarFMA(a, a);
        var gpu = FmaArray.gpuFMA(bufferA, bufferA, bufferC);
        gpu.intoArray(resGPU);

        for (var i = 0; i < SIZE; i++) {
            assertEquals(result[i], resGPU[i], 0.001f);
        }
    }

    @Test
    void FMAvsFMAGPU512() {
        final int SIZE = 512;

        var a = GeneratorHelpers.initFloatArray(SIZE);
        var resGPU = new float[SIZE];
        var bufferA = SVMBuffer.fromArray(SPECIES_SVM, a);
        var bufferC = SVMBuffer.zero(SPECIES_SVM, bufferA.length, bufferA.type);

        var result = FmaArray.scalarFMA(a, a);
        var gpu = FmaArray.gpuFMA(bufferA, bufferA, bufferC);
        gpu.intoArray(resGPU);

        for (var i = 0; i < SIZE; i++) {
            assertEquals(result[i], resGPU[i], 0.001f);
        }
    }

    @Test
    void FMAvsFMAGPU513() {
        final int SIZE = 513;

        var a = GeneratorHelpers.initFloatArray(SIZE);
        var resGPU = new float[SIZE];
        var bufferA = SVMBuffer.fromArray(SPECIES_SVM, a);
        var bufferC = SVMBuffer.zero(SPECIES_SVM, bufferA.length, bufferA.type);

        var result = FmaArray.scalarFMA(a, a);
        var gpu = FmaArray.gpuFMA(bufferA, bufferA, bufferC);
        gpu.intoArray(resGPU);

        for (var i = 0; i < SIZE; i++) {
            assertEquals(result[i], resGPU[i], 0.001f);
        }
    }
}
