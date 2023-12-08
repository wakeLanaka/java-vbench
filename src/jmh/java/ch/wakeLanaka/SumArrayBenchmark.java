// package ch.wakeLanaka;

// import org.openjdk.jmh.annotations.*;
// import org.openjdk.jmh.infra.Blackhole;

// import java.util.Random;
// import java.util.concurrent.TimeUnit;

// import jdk.incubator.vector.SVMBuffer;
// import jdk.incubator.vector.GPUInformation;


// @State(Scope.Benchmark)
// @Fork(jvmArgsPrepend = {"--add-modules=jdk.incubator.vector",
//     "-XX:-TieredCompilation",
//     "-XX:+UseVectorCmov",
//     "-XX:+UseCMoveUnconditionally",
//     "-Djdk.incubator.vector.VECTOR_ACCESS_OOB_CHECK=0"})

// public class SumArrayBenchmark {


//     @State(Scope.Benchmark)
//     public static class BenchmarkState {
//         private static final GPUInformation SPECIES = SVMBuffer.SPECIES_PREFERRED;

//         @Param({"15", "255", "4095", "65535", "1048575", "16777215", "268435455", "346435455"})
//         // @Param({"400000000"})
//         public int LENGTH;
//         public float[] a;
//         public float[] b;
//         public SVMBuffer bufferA;
//         public SVMBuffer bufferB;

//         @Setup(Level.Trial)
//         public void init() {
//             this.a = GeneratorHelpers.initFloatArray(LENGTH);
//             this.b = GeneratorHelpers.initFloatArray(LENGTH);
//             this.bufferA = SVMBuffer.fromArray(SPECIES, this.a);
//             this.bufferB = SVMBuffer.fromArray(SPECIES, this.b);
//             System.out.println("init");
//         }

//         @TearDown(Level.Trial)
//         public void doTearDown() {
//             this.bufferA.releaseSVMBuffer();
//             this.bufferB.releaseSVMBuffer();
//             System.out.println("Tear Down");
//         }
//     }


//     @Fork(value = 1)
//     @Benchmark
//     @BenchmarkMode(Mode.AverageTime)
//     public void arraySumScalar(Blackhole bh, BenchmarkState state) {
//         bh.consume(SumArray.scalarComputation(state.a, state.b));
//     }

//     @Fork(value = 1)
//     @Benchmark
//     @BenchmarkMode(Mode.AverageTime)
//     public void arraySumZeroCopy(Blackhole bh, BenchmarkState state) {
//         bh.consume(SumArray.gpuZeroCopyAdd(state.a, state.b));
//     }

//     @Fork(value = 1)
//     @Benchmark
//     @BenchmarkMode(Mode.AverageTime)
//     public void arraySVMSum(Blackhole bh, BenchmarkState state) {
//         bh.consume(SumArray.gpuSVMAddition(state.bufferA, state.bufferB));
//     }

//     // @Fork(value = 1)
//     // @Benchmark
//     // @BenchmarkMode(Mode.AverageTime)
//     // public void arraySVMSum(Blackhole bh, BenchmarkState state) {
//     //     bh.consume(SumArray.gpuSVMAddition(state.bufferA, state.bufferB));
//     // }
// }
