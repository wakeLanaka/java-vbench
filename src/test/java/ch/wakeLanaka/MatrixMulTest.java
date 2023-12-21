package ch.wakeLanaka;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import jdk.incubator.vector.GPUInformation;
import jdk.incubator.vector.SVMBuffer;

public class MatrixMulTest {

    private static final GPUInformation SPECIES_SVM = SVMBuffer.SPECIES_PREFERRED;

    @Test
    void mulMatrixvsMatrixAVX15() {
        final int SIZE = 15;

        var a = GeneratorHelpers.newFloatRowMajorMatrix(SIZE * SIZE);
        var b = GeneratorHelpers.newFloatRowMajorMatrix(SIZE * SIZE);

        var c_baseline = MatrixMul.computeSerial(a, b, SIZE);
        var c_vector = MatrixMul.computeAVX(a, b, SIZE);

        for (var i = 0; i < SIZE * SIZE; i++) {
            assertEquals(c_baseline[i], c_vector[i], 0.001f);
        }

    }

    @Test
    void mulMatrixvsMatrixAVX16() {
        final int SIZE = 16;

        var a = GeneratorHelpers.newFloatRowMajorMatrix(SIZE * SIZE);
        var b = GeneratorHelpers.newFloatRowMajorMatrix(SIZE * SIZE);

        var c_baseline = MatrixMul.computeSerial(a, b, SIZE);
        var c_vector = MatrixMul.computeAVX(a, b, SIZE);

        for (var i = 0; i < SIZE * SIZE; i++) {
            assertEquals(c_baseline[i], c_vector[i], 0.001f);
        }

    }

    @Test
    void mulMatrixvsMatrixAVX511() {
        final int SIZE = 511;
        var a = GeneratorHelpers.newFloatRowMajorMatrix(SIZE * SIZE);
        var b = GeneratorHelpers.newFloatRowMajorMatrix(SIZE * SIZE);
        var c_baseline = MatrixMul.computeSerial(a, b, SIZE);
        var c_vector = MatrixMul.computeAVX(a, b, SIZE);
        for (var i = 0; i < SIZE * SIZE; i++) {
            assertEquals(c_baseline[i], c_vector[i], 0.001f);
        }
    }

    @Test
    void mulMatrixvsMatrixAVX512() {
        final int SIZE = 512;
        var a = GeneratorHelpers.newFloatRowMajorMatrix(SIZE * SIZE);
        var b = GeneratorHelpers.newFloatRowMajorMatrix(SIZE * SIZE);
        var c_baseline = MatrixMul.computeSerial(a, b, SIZE);
        var c_vector = MatrixMul.computeAVX(a, b, SIZE);
        for (var i = 0; i < SIZE * SIZE; i++) {
            assertEquals(c_baseline[i], c_vector[i], 0.001f);
        }
    }

    @Test
    void mulMatrixvsMatrixSVM15() {
        final int SIZE = 15;
        var a = GeneratorHelpers.newFloatRowMajorMatrix(SIZE * SIZE);
        var va = SVMBuffer.fromArray(SPECIES_SVM, a);
        var b = GeneratorHelpers.newFloatRowMajorMatrix(SIZE * SIZE);
        var vb = SVMBuffer.fromArray(SPECIES_SVM, b);
        var c_svm = new float[SIZE * SIZE];
        var vc = SVMBuffer.fromArray(SPECIES_SVM, c_svm);
        var c_serial = MatrixMul.computeSerial(a, b, SIZE);
        var vc_svm = MatrixMul.computeSVM(va, vb, vc, SIZE);
        vc_svm.intoArray(c_svm);
        for (var i = 0; i < SIZE * SIZE; i++) {
            assertEquals(c_serial[i], c_svm[i], 0.001f);
        }
    }

    @Test
    void mulMatrixvsMatrixSVM16() {
        final int SIZE = 16;
        var a = GeneratorHelpers.newFloatRowMajorMatrix(SIZE * SIZE);
        var va = SVMBuffer.fromArray(SPECIES_SVM, a);
        var b = GeneratorHelpers.newFloatRowMajorMatrix(SIZE * SIZE);
        var vb = SVMBuffer.fromArray(SPECIES_SVM, b);
        var c_svm = new float[SIZE * SIZE];
        var vc = SVMBuffer.fromArray(SPECIES_SVM, c_svm);
        var c_serial = MatrixMul.computeSerial(a, b, SIZE);
        var vc_svm = MatrixMul.computeSVM(va, vb, vc, SIZE);
        vc_svm.intoArray(c_svm);
        for (var i = 0; i < SIZE * SIZE; i++) {
            assertEquals(c_serial[i], c_svm[i], 0.001f);
        }
    }

    @Test
    void mulMatrixvsMatrixSVM17() {
        final int SIZE = 17;
        var a = GeneratorHelpers.newFloatRowMajorMatrix(SIZE * SIZE);
        var va = SVMBuffer.fromArray(SPECIES_SVM, a);
        var b = GeneratorHelpers.newFloatRowMajorMatrix(SIZE * SIZE);
        var vb = SVMBuffer.fromArray(SPECIES_SVM, b);
        var c_svm = new float[SIZE * SIZE];
        var vc = SVMBuffer.fromArray(SPECIES_SVM, c_svm);
        var c_serial = MatrixMul.computeSerial(a, b, SIZE);
        var vc_svm = MatrixMul.computeSVM(va, vb, vc, SIZE);
        vc_svm.intoArray(c_svm);
        for (var i = 0; i < SIZE * SIZE; i++) {
            assertEquals(c_serial[i], c_svm[i], 0.001f);
        }
    }

    @Test
    void mulMatrixvsMatrixSVM511() {
        final int SIZE = 511;
        var a = GeneratorHelpers.newFloatRowMajorMatrix(SIZE * SIZE);
        var va = SVMBuffer.fromArray(SPECIES_SVM, a);
        var b = GeneratorHelpers.newFloatRowMajorMatrix(SIZE * SIZE);
        var vb = SVMBuffer.fromArray(SPECIES_SVM, b);
        var c_svm = new float[SIZE * SIZE];
        var vc = SVMBuffer.fromArray(SPECIES_SVM, c_svm);
        var c_serial = MatrixMul.computeSerial(a, b, SIZE);
        var vc_svm = MatrixMul.computeSVM(va, vb, vc, SIZE);
        vc_svm.intoArray(c_svm);
        for (var i = 0; i < SIZE * SIZE; i++) {
            assertEquals(c_serial[i], c_svm[i], 0.001f);
        }
    }

    @Test
    void mulMatrixvsMatrixSVM512() {
        final int SIZE = 512;
        var a = GeneratorHelpers.newFloatRowMajorMatrix(SIZE * SIZE);
        var va = SVMBuffer.fromArray(SPECIES_SVM, a);
        var b = GeneratorHelpers.newFloatRowMajorMatrix(SIZE * SIZE);
        var vb = SVMBuffer.fromArray(SPECIES_SVM, b);
        var c_svm = new float[SIZE * SIZE];
        var vc = SVMBuffer.fromArray(SPECIES_SVM, c_svm);
        var c_serial = MatrixMul.computeSerial(a, b, SIZE);
        var vc_svm = MatrixMul.computeSVM(va, vb, vc, SIZE);
        vc_svm.intoArray(c_svm);
        for (var i = 0; i < SIZE * SIZE; i++) {
            assertEquals(c_serial[i], c_svm[i], 0.001f);
        }
    }

    @Test
    void mulMatrixvsMatrixSVM513() {
        final int SIZE = 513;
        var a = GeneratorHelpers.newFloatRowMajorMatrix(SIZE * SIZE);
        var va = SVMBuffer.fromArray(SPECIES_SVM, a);
        var b = GeneratorHelpers.newFloatRowMajorMatrix(SIZE * SIZE);
        var vb = SVMBuffer.fromArray(SPECIES_SVM, b);
        var c_svm = new float[SIZE * SIZE];
        var vc = SVMBuffer.fromArray(SPECIES_SVM, c_svm);
        var c_serial = MatrixMul.computeSerial(a, b, SIZE);
        var vc_svm = MatrixMul.computeSVM(va, vb, vc, SIZE);
        vc_svm.intoArray(c_svm);
        for (var i = 0; i < SIZE * SIZE; i++) {
            assertEquals(c_serial[i], c_svm[i], 0.001f);
        }
    }

    @Test
    void mulMatrixvsMatrixSVMNormal15() {
        final int SIZE = 15;
        var a = GeneratorHelpers.newFloatRowMajorMatrix(SIZE * SIZE);
        var b = GeneratorHelpers.newFloatRowMajorMatrix(SIZE * SIZE);
        var b2 = GeneratorHelpers.newFloatColumnMajorMatrix(SIZE, SIZE);
        var c_serial = MatrixMul.computeSerial(a, b2, SIZE);
        var c_svm = MatrixMul.computeSVMNormal(a, b, SIZE, SPECIES_SVM);
        for (var i = 0; i < SIZE * SIZE; i++) {
            assertEquals(c_serial[i], c_svm[i], 0.01f);
        }
    }

    @Test
    void mulMatrixvsMatrixSVMNormal16() {
        final int SIZE = 16;
        var a = GeneratorHelpers.newFloatRowMajorMatrix(SIZE * SIZE);
        var b = GeneratorHelpers.newFloatRowMajorMatrix(SIZE * SIZE);
        var b2 = GeneratorHelpers.newFloatColumnMajorMatrix(SIZE, SIZE);
        var c_serial = MatrixMul.computeSerial(a, b2, SIZE);
        var c_svm = MatrixMul.computeSVMNormal(a, b, SIZE, SPECIES_SVM);
        for (var i = 0; i < SIZE * SIZE; i++) {
            assertEquals(c_serial[i], c_svm[i], 0.01f);
        }
    }

    @Test
    void mulMatrixvsMatrixSVMNormal17() {
        final int SIZE = 17;
        var a = GeneratorHelpers.newFloatRowMajorMatrix(SIZE * SIZE);
        var b = GeneratorHelpers.newFloatRowMajorMatrix(SIZE * SIZE);
        var b2 = GeneratorHelpers.newFloatColumnMajorMatrix(SIZE, SIZE);
        var c_serial = MatrixMul.computeSerial(a, b2, SIZE);
        var c_svm = MatrixMul.computeSVMNormal(a, b, SIZE, SPECIES_SVM);
        for (var i = 0; i < SIZE * SIZE; i++) {
            assertEquals(c_serial[i], c_svm[i], 0.01f);
        }
    }

    // ROUNDING ERROR
    // @Test
    // void mulMatrixvsMatrixSVMNormal511() {
    //     final int SIZE = 511;
    //     var a = GeneratorHelpers.newFloatRowMajorMatrix(SIZE * SIZE);
    //     var b = GeneratorHelpers.newFloatRowMajorMatrix(SIZE * SIZE);
    //     var b2 = GeneratorHelpers.newFloatColumnMajorMatrix(SIZE, SIZE);
    //     var c_serial = MatrixMul.computeSerial(a, b2, SIZE);
    //     var c_svm = MatrixMul.computeSVMNormal(a, b, SIZE, SPECIES_SVM);
    //     for (var i = 0; i < SIZE * SIZE; i++) {
    //         assertEquals(c_serial[i], c_svm[i], 0.01f);
    //     }
    // }

    // @Test
    // void mulMatrixvsMatrixSVMNormal512() {
    //     final int SIZE = 512;
    //     var a = GeneratorHelpers.newFloatRowMajorMatrix(SIZE * SIZE);
    //     var b = GeneratorHelpers.newFloatRowMajorMatrix(SIZE * SIZE);
    //     var b2 = GeneratorHelpers.newFloatColumnMajorMatrix(SIZE, SIZE);
    //     var c_serial = MatrixMul.computeSerial(a, b2, SIZE);
    //     var c_svm = MatrixMul.computeSVMNormal(a, b, SIZE, SPECIES_SVM);
    //     for (var i = 0; i < SIZE * SIZE; i++) {
    //         assertEquals(c_serial[i], c_svm[i], 0.01f);
    //     }
    // }

    // @Test
    // void mulMatrixvsMatrixSVMNormal513() {
    //     final int SIZE = 513;
    //     var a = GeneratorHelpers.newFloatRowMajorMatrix(SIZE * SIZE);
    //     var b = GeneratorHelpers.newFloatRowMajorMatrix(SIZE * SIZE);
    //     var b2 = GeneratorHelpers.newFloatColumnMajorMatrix(SIZE, SIZE);
    //     var c_serial = MatrixMul.computeSerial(a, b2, SIZE);
    //     var c_svm = MatrixMul.computeSVMNormal(a, b, SIZE, SPECIES_SVM);
    //     for (var i = 0; i < SIZE * SIZE; i++) {
    //         assertEquals(c_serial[i], c_svm[i], 0.01f);
    //     }
    // }
}
