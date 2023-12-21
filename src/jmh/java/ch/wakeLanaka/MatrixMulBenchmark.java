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
        @Param({"64" ,"1024", "4096", "16384", "32768", "65536"})
        int size;

        private float[] left;
        private float[] right;
        private float[] result;
        private SVMBuffer leftBuf;
        private SVMBuffer rightBuf;
        private SVMBuffer resultBuf;

        @Setup(Level.Iteration)
        public void init() {
            this.left = newFloatRowMajorMatrix(size * size);
            this.right = newFloatRowMajorMatrix(size * size);
            this.leftBuf = SVMBuffer.fromArray(SPECIES_SVM, this.left);
            this.rightBuf = SVMBuffer.fromArray(SPECIES_SVM, this.right);
            this.result = new float[size * size];
            this.resultBuf = SVMBuffer.fromArray(SPECIES_SVM, result);
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

    // @Benchmark
    // public void MatrixMulSVMNormal(Blackhole bh, BenchmarkSetup state){
    //     bh.consume(MatrixMul.computeSVMNormal(state.left, state.right, state.size, SPECIES_SVM));
    // }

    // @Benchmark
    // public void MatrixMulSVMWithCopy(Blackhole bh, BenchmarkSetup state){
    //     var vleft = SVMBuffer.fromArray(SPECIES_SVM, state.left);
    //     var vright = SVMBuffer.fromArray(SPECIES_SVM, state.right);
    //     var vresult = SVMBuffer.fromArray(SPECIES_SVM, state.result);
    //     bh.consume(MatrixMul.computeSVM(vleft, vright, vresult, state.size));
    //     vresult.intoArray(state.result);
    // }


    // @Benchmark
    // public void MatrixMulSVMRange(Blackhole bh, BenchmarkSetup state){
    //     bh.consume(MatrixMul.computeSVMRange(state.leftBuf, state.rightBuf, state.size));
    // }
}
