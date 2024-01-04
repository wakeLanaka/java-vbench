package ch.wakeLanaka;

import org.openjdk.jmh.annotations.*;
import java.util.Random;

import org.openjdk.jmh.infra.Blackhole;
import jdk.incubator.vector.SVMBuffer;
import jdk.incubator.vector.GPUInformation;

public class BlackScholesBenchmark {
    private static final GPUInformation SPECIES_SVM = SVMBuffer.SPECIES_PREFERRED;

    @State(Scope.Thread)
    public static class BenchmarkSetup {

        @Param({"524288", "1048576", "2097152", "4194304", "8388608"})
        public int size;

        float[] s0;
        float[] x;
        float[] t;
        float[] call;
        float[] put;
        SVMBuffer s0Buf;
        SVMBuffer xBuf;
        SVMBuffer tBuf;
        SVMBuffer callBuf;
        SVMBuffer putBuf;
        SVMBuffer sigBuf;
        SVMBuffer rBuf;
        SVMBuffer negrBuf;

        float r;
        float sig;
        Random rand;

        float randFloat(float low, float high) {
            float val = rand.nextFloat();
            return (1.0f - val) * low + val * high;
        }

        float[] fillRandom(float low, float high) {
            float[] array = new float[size];
            for (int i = 0; i < array.length; i++) {
                array[i] = randFloat(low, high);
            }
            return array;
        }

        @Setup(Level.Trial)
        public void doSetup() {
            rand = new Random();
            s0 = fillRandom(5.0f, 30.0f);
            x  = fillRandom(1.0f, 100.0f);
            t  = fillRandom(0.25f, 10.0f);

            r = 0.02f;
            sig = 0.30f;

            call = new float[size];
            put = new float[size];

            rBuf = SVMBuffer.broadcast(SPECIES_SVM, r, size);
            sigBuf = SVMBuffer.broadcast(SPECIES_SVM, sig, size);
            negrBuf = SVMBuffer.broadcast(SPECIES_SVM, -r, size);
            s0Buf = SVMBuffer.fromArray(SPECIES_SVM, s0);
            xBuf = SVMBuffer.fromArray(SPECIES_SVM, x);
            tBuf = SVMBuffer.fromArray(SPECIES_SVM, t);
            callBuf = SVMBuffer.fromArray(SPECIES_SVM, call);
            putBuf = SVMBuffer.fromArray(SPECIES_SVM, put);
        }

        @TearDown(Level.Trial)
        public void doTearDown() {
            call = new float[size]; 
            put = new float[size]; 

            s0Buf.releaseSVMBuffer();
            xBuf.releaseSVMBuffer();
            tBuf.releaseSVMBuffer();
            callBuf.releaseSVMBuffer();
            putBuf.releaseSVMBuffer();
            sigBuf.releaseSVMBuffer();
            rBuf.releaseSVMBuffer();
            negrBuf.releaseSVMBuffer();
        }
    }

    @Benchmark
    @BenchmarkMode(Mode.AverageTime)
    @Fork(1)
    public void blackScholesSerial(BenchmarkSetup state){
        BlackScholes.computeSerial(state.sig, state.r, state.x, state.call, state.put, state.t, state.s0, state.size, 0);
    }

    @Benchmark
    @BenchmarkMode(Mode.AverageTime)
    @Fork(1)
    public void blackScholesAVX(BenchmarkSetup state){
        int offset = BlackScholes.computeAVX(state.sig, state.r, state.x, state.call, state.put, state.t, state.s0);
        for (int i = offset; i < state.size; i++){
            BlackScholes.computeSerial(state.sig, state.r, state.x, state.call, state.put, state.t, state.s0, state.size, offset);
        }
    }

    @Benchmark
    @BenchmarkMode(Mode.AverageTime)
    public void blackScholesSVM(BenchmarkSetup state){
        BlackScholes.computeSVM(state.sig, state.r, state.xBuf, state.call, state.put, state.tBuf, state.s0Buf);
    }

    @Benchmark
    @BenchmarkMode(Mode.AverageTime)
    public void blackScholesSVMWithCopy(BenchmarkSetup state){
        var xBuf = SVMBuffer.fromArray(SPECIES_SVM, state.x);
        var tBuf = SVMBuffer.fromArray(SPECIES_SVM, state.t);
        var s0Buf = SVMBuffer.fromArray(SPECIES_SVM, state.s0);

        BlackScholes.computeSVM(state.sig, state.r, xBuf, state.call, state.put, tBuf, s0Buf);

        xBuf.releaseSVMBuffer();
        tBuf.releaseSVMBuffer();
        s0Buf.releaseSVMBuffer();
    }

//     // @Benchmark
//     // @BenchmarkMode(Mode.AverageTime)
//     // public void blackScholesKernelBuilderWithCopy(BenchmarkSetup state){
//     //     var vx = SVMBuffer.fromArray(SPECIES_SVM, state.x);
//     //     var vcall = SVMBuffer.fromArray(SPECIES_SVM, state.call);
//     //     var vput = SVMBuffer.fromArray(SPECIES_SVM, state.put);
//     //     var vt = SVMBuffer.fromArray(SPECIES_SVM, state.t);
//     //     var vs0 = SVMBuffer.fromArray(SPECIES_SVM, state.s0);

//     //     BlackScholes.computeKernelBuilder(state.sig, state.r, vx, vcall, vput, vt, vs0);

//     //     vcall.intoArray(state.call);
//     //     vput.intoArray(state.put);
//     // }

//     // @Benchmark
//     // @BenchmarkMode(Mode.AverageTime)
//     // public void blackScholesKernelBuilder(BenchmarkSetup state){
//     //     BlackScholes.computeKernelBuilder(state.sig, state.r, state.xBuf, state.callBuf, state.putBuf, state.tBuf, state.s0Buf);
//     // }
}
