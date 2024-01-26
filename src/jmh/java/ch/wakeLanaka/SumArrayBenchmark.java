package ch.wakeLanaka;

import org.openjdk.jmh.annotations.*;
import org.openjdk.jmh.infra.Blackhole;

import java.util.Random;
import java.util.concurrent.TimeUnit;

import jdk.incubator.vector.SVMBuffer;
import jdk.incubator.vector.GPUInformation;


@State(Scope.Benchmark)

public class SumArrayBenchmark {


    @State(Scope.Benchmark)
    public static class BenchmarkState {
        private static final GPUInformation SPECIES = SVMBuffer.SPECIES_PREFERRED;

        @Param({"15", "255", "4095", "65535", "1048575", "16777215", "268435455", "346435455"})
        public int size;
        public float[] a;
        public SVMBuffer bufferA;
        public SVMBuffer bufferB;

        @Setup(Level.Invocation)
        public void init() {
            this.a = GeneratorHelpers.initFloatArray(size);
            this.bufferA = SVMBuffer.fromArray(SPECIES, this.a);
            this.bufferB = SVMBuffer.fromArray(SPECIES, this.a);
        }

        @TearDown(Level.Invocation)
        public void doTearDown() {
            this.bufferA.releaseSVMBuffer();
            this.bufferB.releaseSVMBuffer();
        }
    }

    @Benchmark
    @BenchmarkMode(Mode.AverageTime)
    public void arraySumScalar(Blackhole bh, BenchmarkState state) {
        bh.consume(SumArray.scalarComputation(state.a, state.a));
    }

    @Benchmark
    @BenchmarkMode(Mode.AverageTime)
    public void avx(Blackhole bh, BenchmarkState state) {
        bh.consume(SumArray.vectorComputation(state.a, state.a));
    }

    @Benchmark
    @BenchmarkMode(Mode.AverageTime)
    public void arraySVMSum(Blackhole bh, BenchmarkState state) {
        var bufferC = SumArray.gpuSVMAddition(state.bufferA, state.bufferB);
        bufferC.releaseSVMBuffer();
    }

    @Benchmark
    @BenchmarkMode(Mode.AverageTime)
    public void arraySVMCopySum(Blackhole bh, BenchmarkState state) {
        bh.consume(SumArray.gpuSVMCopyAddition(state.a, state.a));
    }
}
