package ch.wakeLanaka;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import jdk.incubator.vector.GPUInformation;
import jdk.incubator.vector.SVMBuffer;

public class SumArrayTest {

    private static final GPUInformation SPECIES_SVM = SVMBuffer.SPECIES_PREFERRED;

    @Test
    void SUMvsSUMAVX15() {
        final int SIZE = 15;

        var a = GeneratorHelpers.initFloatArray(SIZE);

        var result = SumArray.scalarComputation(a, a);
        var avx = SumArray.vectorComputation(a, a);

        for (var i = 0; i < SIZE; i++) {
            assertEquals(result[i], avx[i], 0.001f);
        }
    }

    @Test
    void SUMvsSUMAVX16() {
        final int SIZE = 16;

        var a = GeneratorHelpers.initFloatArray(SIZE);

        var result = SumArray.scalarComputation(a, a);
        var avx = SumArray.vectorComputation(a, a);

        for (var i = 0; i < SIZE; i++) {
            assertEquals(result[i], avx[i], 0.001f);
        }
    }

    @Test
    void SUMvsSUMAVX17() {
        final int SIZE = 17;

        var a = GeneratorHelpers.initFloatArray(SIZE);

        var result = SumArray.scalarComputation(a, a);
        var avx = SumArray.vectorComputation(a, a);

        for (var i = 0; i < SIZE; i++) {
            assertEquals(result[i], avx[i], 0.001f);
        }
    }

    @Test
    void SUMvsSUMAVX511() {
        final int SIZE = 511;

        var a = GeneratorHelpers.initFloatArray(SIZE);

        var result = SumArray.scalarComputation(a, a);
        var avx = SumArray.vectorComputation(a, a);

        for (var i = 0; i < SIZE; i++) {
            assertEquals(result[i], avx[i], 0.001f);
        }
    }

    @Test
    void SUMvsSUMAVX512() {
        final int SIZE = 512;

        var a = GeneratorHelpers.initFloatArray(SIZE);

        var result = SumArray.scalarComputation(a, a);
        var avx = SumArray.vectorComputation(a, a);

        for (var i = 0; i < SIZE; i++) {
            assertEquals(result[i], avx[i], 0.001f);
        }
    }

    @Test
    void SUMvsSUMAVX513() {
        final int SIZE = 513;

        var a = GeneratorHelpers.initFloatArray(SIZE);

        var result = SumArray.scalarComputation(a, a);
        var avx = SumArray.vectorComputation(a, a);

        for (var i = 0; i < SIZE; i++) {
            assertEquals(result[i], avx[i], 0.001f);
        }
    }

    @Test
    void SUMvsSUMGPUCopy15() {
        final int SIZE = 15;

        var a = GeneratorHelpers.initFloatArray(SIZE);

        var result = SumArray.scalarComputation(a, a);
        var svm = SumArray.gpuSVMCopyAddition(a, a);

        for (var i = 0; i < SIZE; i++) {
            assertEquals(result[i], svm[i], 0.001f);
        }
    }

    @Test
    void SUMvsSUMGPUCopy16() {
        final int SIZE = 16;

        var a = GeneratorHelpers.initFloatArray(SIZE);

        var result = SumArray.scalarComputation(a, a);
        var svm = SumArray.gpuSVMCopyAddition(a, a);

        for (var i = 0; i < SIZE; i++) {
            assertEquals(result[i], svm[i], 0.001f);
        }
    }

    @Test
    void SUMvsSUMGPUCopy17() {
        final int SIZE = 17;

        var a = GeneratorHelpers.initFloatArray(SIZE);

        var result = SumArray.scalarComputation(a, a);
        var svm = SumArray.gpuSVMCopyAddition(a, a);

        for (var i = 0; i < SIZE; i++) {
            assertEquals(result[i], svm[i], 0.001f);
        }
    }

    @Test
    void SUMvsSUMGPUCopy511() {
        final int SIZE = 511;

        var a = GeneratorHelpers.initFloatArray(SIZE);

        var result = SumArray.scalarComputation(a, a);
        var svm = SumArray.gpuSVMCopyAddition(a, a);

        for (var i = 0; i < SIZE; i++) {
            assertEquals(result[i], svm[i], 0.001f);
        }
    }

    @Test
    void SUMvsSUMGPUCopy512() {
        final int SIZE = 512;

        var a = GeneratorHelpers.initFloatArray(SIZE);

        var result = SumArray.scalarComputation(a, a);
        var svm = SumArray.gpuSVMCopyAddition(a, a);

        for (var i = 0; i < SIZE; i++) {
            assertEquals(result[i], svm[i], 0.001f);
        }
    }

    @Test
    void SUMvsSUMGPUCopy513() {
        final int SIZE = 513;

        var a = GeneratorHelpers.initFloatArray(SIZE);

        var result = SumArray.scalarComputation(a, a);
        var svm = SumArray.gpuSVMCopyAddition(a, a);

        for (var i = 0; i < SIZE; i++) {
            assertEquals(result[i], svm[i], 0.001f);
        }
    }


    @Test
    void SUMvsSUMGPU15() {
        final int SIZE = 15;

        var a = GeneratorHelpers.initFloatArray(SIZE);
        var bufferA = SVMBuffer.fromArray(SPECIES_SVM, a);
        var bufferB = SVMBuffer.fromArray(SPECIES_SVM, a);
        var c = new float[a.length];

        var result = SumArray.scalarComputation(a, a);
        var svm = SumArray.gpuSVMAddition(bufferA, bufferB);
        svm.intoArray(c);

        for (var i = 0; i < SIZE; i++) {
            assertEquals(result[i], c[i], 0.001f);
        }
    }

    @Test
    void SUMvsSUMGPU16() {
        final int SIZE = 16;

        var a = GeneratorHelpers.initFloatArray(SIZE);
        var bufferA = SVMBuffer.fromArray(SPECIES_SVM, a);
        var bufferB = SVMBuffer.fromArray(SPECIES_SVM, a);
        var c = new float[a.length];

        var result = SumArray.scalarComputation(a, a);
        var svm = SumArray.gpuSVMAddition(bufferA, bufferB);
        svm.intoArray(c);

        for (var i = 0; i < SIZE; i++) {
            assertEquals(result[i], c[i], 0.001f);
        }
    }

    @Test
    void SUMvsSUMGPU17() {
        final int SIZE = 17;

        var a = GeneratorHelpers.initFloatArray(SIZE);
        var bufferA = SVMBuffer.fromArray(SPECIES_SVM, a);
        var bufferB = SVMBuffer.fromArray(SPECIES_SVM, a);
        var c = new float[a.length];

        var result = SumArray.scalarComputation(a, a);
        var svm = SumArray.gpuSVMAddition(bufferA, bufferB);
        svm.intoArray(c);

        for (var i = 0; i < SIZE; i++) {
            assertEquals(result[i], c[i], 0.001f);
        }
    }

    @Test
    void SUMvsSUMGPU511() {
        final int SIZE = 511;

        var a = GeneratorHelpers.initFloatArray(SIZE);
        var bufferA = SVMBuffer.fromArray(SPECIES_SVM, a);
        var bufferB = SVMBuffer.fromArray(SPECIES_SVM, a);
        var c = new float[a.length];

        var result = SumArray.scalarComputation(a, a);
        var svm = SumArray.gpuSVMAddition(bufferA, bufferB);
        svm.intoArray(c);

        for (var i = 0; i < SIZE; i++) {
            assertEquals(result[i], c[i], 0.001f);
        }
    }

    @Test
    void SUMvsSUMGPU512() {
        final int SIZE = 512;

        var a = GeneratorHelpers.initFloatArray(SIZE);
        var bufferA = SVMBuffer.fromArray(SPECIES_SVM, a);
        var bufferB = SVMBuffer.fromArray(SPECIES_SVM, a);
        var c = new float[a.length];

        var result = SumArray.scalarComputation(a, a);
        var svm = SumArray.gpuSVMAddition(bufferA, bufferB);
        svm.intoArray(c);

        for (var i = 0; i < SIZE; i++) {
            assertEquals(result[i], c[i], 0.001f);
        }
    }

    @Test
    void SUMvsSUMGPU513() {
        final int SIZE = 513;

        var a = GeneratorHelpers.initFloatArray(SIZE);
        var bufferA = SVMBuffer.fromArray(SPECIES_SVM, a);
        var bufferB = SVMBuffer.fromArray(SPECIES_SVM, a);
        var c = new float[a.length];

        var result = SumArray.scalarComputation(a, a);
        var svm = SumArray.gpuSVMAddition(bufferA, bufferB);
        svm.intoArray(c);

        for (var i = 0; i < SIZE; i++) {
            assertEquals(result[i], c[i], 0.001f);
        }
    }
}
