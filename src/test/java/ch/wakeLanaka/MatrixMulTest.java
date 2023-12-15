package ch.wakeLanaka;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;
import jdk.incubator.vector.GPUInformation;
import jdk.incubator.vector.SVMBuffer;

public class MatrixMulTest {

    private static final GPUInformation SPECIES_SVM = SVMBuffer.SPECIES_PREFERRED;

    // @Test
    // void mulMatrixvsMatrixAVX() {
    //     // Prime Number, that doesn't make the registers align by accident!
    //     final int SIZE = 512;

    //     var a = GeneratorHelpers.newFloatRowMajorMatrix(SIZE * SIZE);
    //     var b = GeneratorHelpers.newFloatRowMajorMatrix(SIZE * SIZE);

    //     var c_baseline = MatrixMul.computeSerial(a, b, SIZE);
    //     var c_vector = MatrixMul.computeAVX(a, b, SIZE);

    //     for (var i = 0; i < SIZE * SIZE; i++) {
    //         assertEquals(c_baseline[i], c_vector[i], 0.001f);
    //     }

    // }

    // @Test
    // void mulMatrixvsMatrixSVM() {
    //     // Prime Number, that doesn't make the registers align by accident!
    //     final int SIZE = 512;


    //     var a = GeneratorHelpers.iotaFloatRowMajorMatrix(SIZE * SIZE);
    //     var va = SVMBuffer.fromArray(SPECIES_SVM, a);
    //     var b = GeneratorHelpers.iotaFloatRowMajorMatrix(SIZE * SIZE);
    //     var vb = SVMBuffer.fromArray(SPECIES_SVM, b);
    //     var c_svm = new float[SIZE * SIZE];
    //     var vc = SVMBuffer.fromArray(SPECIES_SVM, c_svm);

    //     var c_serial = MatrixMul.computeSerial(a, b, SIZE);
    //     var vc_svm = MatrixMul.computeSVM(va, vb, vc, SIZE);
    //     vc_svm.intoArray(c_svm);


    //     for (var i = 0; i < SIZE * SIZE; i++) {
    //         assertEquals(c_serial[i], c_svm[i], 0.001f);
    //     }
    // }

    // @Test
    // void mulMatrixvsMatrixSVM() {
    //     // Prime Number, that doesn't make the registers align by accident!
    //     final int SIZE = 4;

    //     var a = GeneratorHelpers.iotaFloatRowMajorMatrix(SIZE * SIZE);
    //     var b = GeneratorHelpers.iotaFloatRowMajorMatrix(SIZE * SIZE);
    //     var b2 = GeneratorHelpers.iotaFloatColumnMajorMatrix(SIZE, SIZE);

    //     var c_serial = MatrixMul.computeSerial(a, b2, SIZE);
    //     var c_svm = MatrixMul.computeSVMSlow(a, b, SIZE, SPECIES_SVM);

    //     for (var i = 0; i < SIZE * SIZE; i++) {
    //         System.out.println("------");
    //         System.out.println("serial: " + c_serial[i]);
    //         System.out.println("svm: " + c_svm[i]);
    //         assertEquals(c_serial[i], c_svm[i], 0.001f);
    //     }
    // }

    @Test
    void mulMatrixvsMatrixSVMRange() {
        // Prime Number, that doesn't make the registers align by accident!
        final int SIZE = 4;

        var a = GeneratorHelpers.iotaFloatRowMajorMatrix(SIZE * SIZE);
        var va = SVMBuffer.fromArray(SPECIES_SVM, a);
        var b = GeneratorHelpers.iotaFloatRowMajorMatrix(SIZE * SIZE);
        var vb = SVMBuffer.fromArray(SPECIES_SVM, a);
        var b2 = GeneratorHelpers.iotaFloatColumnMajorMatrix(SIZE, SIZE);

        var c_serial = MatrixMul.computeSerial(a, b2, SIZE);
        var c_svm = MatrixMul.computeSVMRange(va, vb, SIZE);

        for (var i = 0; i < SIZE * SIZE; i++) {
            System.out.println("------");
            System.out.println("serial: " + c_serial[i]);
            System.out.println("svm: " + c_svm[i]);
            assertEquals(c_serial[i], c_svm[i], 0.001f);
        }
    }
}
