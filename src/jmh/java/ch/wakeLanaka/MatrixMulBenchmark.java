package ch.wakeLanaka;

import jdk.incubator.vector.VectorSpecies;
import org.openjdk.jmh.annotations.*;
import org.openjdk.jmh.infra.Blackhole;

import static ch.wakeLanaka.GeneratorHelpers.newFloatRowMajorMatrix;
import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.GPUInformation;
import jdk.incubator.vector.SVMBuffer;

public class MatrixMulBenchmark {
    private static final GPUInformation SPECIES_SVM = SVMBuffer.SPECIES_PREFERRED;

    @State(Scope.Thread)
    public static class BenchmarkSetup {
        @Param({"512", "1024", "2048", "4096"})
        int size;

        private float[] left;
        private float[] right;
        private float[] result;
        private SVMBuffer leftBuf;
        private SVMBuffer rightBuf;
        private SVMBuffer resultBuf;

        @Setup(Level.Trial)
        public void doSetup() {
            left = newFloatRowMajorMatrix(size * size);
            right = newFloatRowMajorMatrix(size * size);
            leftBuf = SVMBuffer.fromArray(SPECIES_SVM, this.left);
            rightBuf = SVMBuffer.fromArray(SPECIES_SVM, this.right);
            result = new float[size * size];
            resultBuf = SVMBuffer.fromArray(SPECIES_SVM, result);
        }

        @TearDown(Level.Trial)
        public void doTearDown(){
            leftBuf.releaseSVMBuffer();
            rightBuf.releaseSVMBuffer();
            resultBuf.releaseSVMBuffer();
        }
    }

    @Benchmark
    public void MatrixMulSerial(Blackhole bh, BenchmarkSetup state) {
        bh.consume(MatrixMul.computeSerial(state.left, state.right, state.size));
    }

    @Benchmark
    public void MatrixMulAVX(Blackhole bh, BenchmarkSetup state) {
        bh.consume(MatrixMul.computeAVX(state.left, state.right, state.size));
    }

    @Benchmark
    public void MatrixMulSVM(Blackhole bh, BenchmarkSetup state){
        bh.consume(MatrixMul.computeSVM(state.leftBuf, state.rightBuf, state.resultBuf, state.size));
    }

    @Benchmark
    public void MatrixMulSVMMatrix(Blackhole bh, BenchmarkSetup state){
        bh.consume(MatrixMul.computeSVMMatrix(state.leftBuf, state.right, state.size));
    }

    // @Benchmark
    // public void MatrixMulSVMWithCopy(Blackhole bh, BenchmarkSetup state){
    //     var vleft = SVMBuffer.fromArray(SPECIES_SVM, state.left);
    //     var vright = SVMBuffer.fromArray(SPECIES_SVM, state.right);
    //     var vresult = SVMBuffer.fromArray(SPECIES_SVM, state.result);
    //     bh.consume(MatrixMul.computeSVM(vleft, vright, vresult, state.size));
    //     vresult.intoArray(state.result);
    //     vleft.releaseSVMBuffer();
    //     vright.releaseSVMBuffer();
    //     vresult.releaseSVMBuffer();
    // }
}
