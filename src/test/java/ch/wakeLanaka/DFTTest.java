// package ch.wakeLanaka;

// import org.junit.jupiter.api.Test;

// import static org.junit.jupiter.api.Assertions.assertEquals;
// import jdk.incubator.vector.GPUInformation;
// import jdk.incubator.vector.SVMBuffer;

// public class DFTTest {

//     private static final GPUInformation SPECIES_SVM = SVMBuffer.SPECIES_PREFERRED;

//     @Test
//     void DFTvsDFTAVX15() {
//         final int SIZE = 15;

//         var inReal = GeneratorHelpers.newFloatRowMajorMatrix(SIZE);
//         var inImag = GeneratorHelpers.newFloatRowMajorMatrix(SIZE);
//         var t = GeneratorHelpers.iotaFloatArray(SIZE);

//         float[] outRealSerial = new float[SIZE];
//         float[] outImagSerial = new float[SIZE];
//         float[] outRealAVX = new float[SIZE];
//         float[] outImagAVX = new float[SIZE];

//         DFT.computeSerial(inReal, outRealSerial, inImag, outImagSerial);
//         DFT.computeAVX(inReal, outRealAVX, inImag, outImagAVX, t);

//         for (var i = 0; i < SIZE; i++) {
//             assertEquals(outRealSerial[i], outRealAVX[i], 0.001f);
//             assertEquals(outImagSerial[i], outImagAVX[i], 0.001f);
//         }
//     }

//     @Test
//     void DFTvsDFTAVX16() {
//         final int SIZE = 16;

//         var inReal = GeneratorHelpers.newFloatRowMajorMatrix(SIZE);
//         var inImag = GeneratorHelpers.newFloatRowMajorMatrix(SIZE);
//         var t = GeneratorHelpers.iotaFloatArray(SIZE);

//         float[] outRealSerial = new float[SIZE];
//         float[] outImagSerial = new float[SIZE];
//         float[] outRealAVX = new float[SIZE];
//         float[] outImagAVX = new float[SIZE];

//         DFT.computeSerial(inReal, outRealSerial, inImag, outImagSerial);
//         DFT.computeAVX(inReal, outRealAVX, inImag, outImagAVX, t);

//         for (var i = 0; i < SIZE; i++) {
//             assertEquals(outRealSerial[i], outRealAVX[i], 0.001f);
//             assertEquals(outImagSerial[i], outImagAVX[i], 0.001f);
//         }
//     }

//     @Test
//     void DFTvsDFTAVX17() {
//         final int SIZE = 17;

//         var inReal = GeneratorHelpers.newFloatRowMajorMatrix(SIZE);
//         var inImag = GeneratorHelpers.newFloatRowMajorMatrix(SIZE);
//         var t = GeneratorHelpers.iotaFloatArray(SIZE);

//         float[] outRealSerial = new float[SIZE];
//         float[] outImagSerial = new float[SIZE];
//         float[] outRealAVX = new float[SIZE];
//         float[] outImagAVX = new float[SIZE];

//         DFT.computeSerial(inReal, outRealSerial, inImag, outImagSerial);
//         DFT.computeAVX(inReal, outRealAVX, inImag, outImagAVX, t);

//         for (var i = 0; i < SIZE; i++) {
//             assertEquals(outRealSerial[i], outRealAVX[i], 0.001f);
//             assertEquals(outImagSerial[i], outImagAVX[i], 0.001f);
//         }
//     }

//     @Test
//     void DFTvsDFTAVX511() {
//         final int SIZE = 511;

//         var inReal = GeneratorHelpers.newFloatRowMajorMatrix(SIZE);
//         var inImag = GeneratorHelpers.newFloatRowMajorMatrix(SIZE);
//         var t = GeneratorHelpers.iotaFloatArray(SIZE);

//         float[] outRealSerial = new float[SIZE];
//         float[] outImagSerial = new float[SIZE];
//         float[] outRealAVX = new float[SIZE];
//         float[] outImagAVX = new float[SIZE];

//         DFT.computeSerial(inReal, outRealSerial, inImag, outImagSerial);
//         DFT.computeAVX(inReal, outRealAVX, inImag, outImagAVX, t);

//         for (var i = 0; i < SIZE; i++) {
//             assertEquals(outRealSerial[i], outRealAVX[i], 10f);
//             assertEquals(outImagSerial[i], outImagAVX[i], 10f);
//         }
//     }

//     @Test
//     void DFTvsDFTAVX512() {
//         final int SIZE = 512;

//         var inReal = GeneratorHelpers.newFloatRowMajorMatrix(SIZE);
//         var inImag = GeneratorHelpers.newFloatRowMajorMatrix(SIZE);
//         var t = GeneratorHelpers.iotaFloatArray(SIZE);

//         float[] outRealSerial = new float[SIZE];
//         float[] outImagSerial = new float[SIZE];
//         float[] outRealAVX = new float[SIZE];
//         float[] outImagAVX = new float[SIZE];

//         DFT.computeSerial(inReal, outRealSerial, inImag, outImagSerial);
//         DFT.computeAVX(inReal, outRealAVX, inImag, outImagAVX, t);

//         for (var i = 0; i < SIZE; i++) {
//             assertEquals(outRealSerial[i], outRealAVX[i], 10f);
//             assertEquals(outImagSerial[i], outImagAVX[i], 10f);
//         }
//     }

//     @Test
//     void DFTvsDFTAVX513() {
//         final int SIZE = 513;

//         var inReal = GeneratorHelpers.newFloatRowMajorMatrix(SIZE);
//         var inImag = GeneratorHelpers.newFloatRowMajorMatrix(SIZE);
//         var t = GeneratorHelpers.iotaFloatArray(SIZE);

//         float[] outRealSerial = new float[SIZE];
//         float[] outImagSerial = new float[SIZE];
//         float[] outRealAVX = new float[SIZE];
//         float[] outImagAVX = new float[SIZE];

//         DFT.computeSerial(inReal, outRealSerial, inImag, outImagSerial);
//         DFT.computeAVX(inReal, outRealAVX, inImag, outImagAVX, t);

//         for (var i = 0; i < SIZE; i++) {
//             assertEquals(outRealSerial[i], outRealAVX[i], 10f);
//             assertEquals(outImagSerial[i], outImagAVX[i], 10f);
//         }
//     }

//     @Test
//     void DFTvsDFTSVM15() {
//         final int SIZE = 15;

//         var inReal = GeneratorHelpers.newFloatRowMajorMatrix(SIZE);
//         var inImag = GeneratorHelpers.newFloatRowMajorMatrix(SIZE);
//         var t = GeneratorHelpers.iotaFloatArray(SIZE);

//         float[] outRealSerial = new float[SIZE];
//         float[] outImagSerial = new float[SIZE];
//         float[] outRealAVX = new float[SIZE];
//         float[] outImagAVX = new float[SIZE];

//         DFT.computeSerial(inReal, outRealSerial, inImag, outImagSerial);
//         DFT.computeAVX(inReal, outRealAVX, inImag, outImagAVX, t);

//         for (var i = 0; i < SIZE; i++) {
//             assertEquals(outRealSerial[i], outRealAVX[i], 0.01f);
//             assertEquals(outImagSerial[i], outImagAVX[i], 0.01f);
//         }
//     }

//     @Test
//     void DFTvsDFTSVM16() {
//         final int SIZE = 16;

//         var inReal = GeneratorHelpers.newFloatRowMajorMatrix(SIZE);
//         var inImag = GeneratorHelpers.newFloatRowMajorMatrix(SIZE);
//         var t = GeneratorHelpers.iotaFloatArray(SIZE);

//         float[] outRealSerial = new float[SIZE];
//         float[] outImagSerial = new float[SIZE];
//         float[] outRealAVX = new float[SIZE];
//         float[] outImagAVX = new float[SIZE];

//         DFT.computeSerial(inReal, outRealSerial, inImag, outImagSerial);
//         DFT.computeAVX(inReal, outRealAVX, inImag, outImagAVX, t);

//         for (var i = 0; i < SIZE; i++) {
//             assertEquals(outRealSerial[i], outRealAVX[i], 0.01f);
//             assertEquals(outImagSerial[i], outImagAVX[i], 0.01f);
//         }
//     }

//     @Test
//     void DFTvsDFTSVM17() {
//         final int SIZE = 17;

//         var inReal = GeneratorHelpers.newFloatRowMajorMatrix(SIZE);
//         var inImag = GeneratorHelpers.newFloatRowMajorMatrix(SIZE);
//         var t = GeneratorHelpers.iotaFloatArray(SIZE);

//         float[] outRealSerial = new float[SIZE];
//         float[] outImagSerial = new float[SIZE];
//         float[] outRealAVX = new float[SIZE];
//         float[] outImagAVX = new float[SIZE];

//         DFT.computeSerial(inReal, outRealSerial, inImag, outImagSerial);
//         DFT.computeAVX(inReal, outRealAVX, inImag, outImagAVX, t);

//         for (var i = 0; i < SIZE; i++) {
//             assertEquals(outRealSerial[i], outRealAVX[i], 0.01f);
//             assertEquals(outImagSerial[i], outImagAVX[i], 0.01f);
//         }
//     }

//     @Test
//     void DFTvsDFTSVM511() {
//         final int SIZE = 511;

//         var inReal = GeneratorHelpers.newFloatRowMajorMatrix(SIZE);
//         var inImag = GeneratorHelpers.newFloatRowMajorMatrix(SIZE);
//         var t = GeneratorHelpers.iotaFloatArray(SIZE);

//         float[] outRealSerial = new float[SIZE];
//         float[] outImagSerial = new float[SIZE];
//         float[] outRealAVX = new float[SIZE];
//         float[] outImagAVX = new float[SIZE];

//         DFT.computeSerial(inReal, outRealSerial, inImag, outImagSerial);
//         DFT.computeAVX(inReal, outRealAVX, inImag, outImagAVX, t);

//         for (var i = 0; i < SIZE; i++) {
//             assertEquals(outRealSerial[i], outRealAVX[i], 10f);
//             assertEquals(outImagSerial[i], outImagAVX[i], 10f);
//         }
//     }

//     @Test
//     void DFTvsDFTSVM512() {
//         final int SIZE = 512;

//         var inReal = GeneratorHelpers.newFloatRowMajorMatrix(SIZE);
//         var inImag = GeneratorHelpers.newFloatRowMajorMatrix(SIZE);
//         var t = GeneratorHelpers.iotaFloatArray(SIZE);

//         float[] outRealSerial = new float[SIZE];
//         float[] outImagSerial = new float[SIZE];
//         float[] outRealAVX = new float[SIZE];
//         float[] outImagAVX = new float[SIZE];

//         DFT.computeSerial(inReal, outRealSerial, inImag, outImagSerial);
//         DFT.computeAVX(inReal, outRealAVX, inImag, outImagAVX, t);

//         for (var i = 0; i < SIZE; i++) {
//             assertEquals(outRealSerial[i], outRealAVX[i], 10f);
//             assertEquals(outImagSerial[i], outImagAVX[i], 10f);
//         }
//     }

//     @Test
//     void DFTvsDFTSVM513() {
//         final int SIZE = 513;

//         var inReal = GeneratorHelpers.newFloatRowMajorMatrix(SIZE);
//         var inImag = GeneratorHelpers.newFloatRowMajorMatrix(SIZE);
//         var t = GeneratorHelpers.iotaFloatArray(SIZE);

//         float[] outRealSerial = new float[SIZE];
//         float[] outImagSerial = new float[SIZE];
//         float[] outRealAVX = new float[SIZE];
//         float[] outImagAVX = new float[SIZE];

//         DFT.computeSerial(inReal, outRealSerial, inImag, outImagSerial);
//         DFT.computeAVX(inReal, outRealAVX, inImag, outImagAVX, t);

//         for (var i = 0; i < SIZE; i++) {
//             assertEquals(outRealSerial[i], outRealAVX[i], 10f);
//             assertEquals(outImagSerial[i], outImagAVX[i], 10f);
//         }
//     }
// }
