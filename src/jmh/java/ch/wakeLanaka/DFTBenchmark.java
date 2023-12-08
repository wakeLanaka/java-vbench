package ch.wakeLanaka;

import org.openjdk.jmh.annotations.*;
import java.util.Random;

import jdk.incubator.vector.SVMBuffer;
import jdk.incubator.vector.GPUInformation;

public class DFTBenchmark {
    private static final GPUInformation SPECIES_SVM = SVMBuffer.SPECIES_PREFERRED;

    @State(Scope.Thread)
    public static class BenchmarkSetup{

        @Param({/* "1024" *//* , "16384", "32768", */ "65536"})
        public int size;
        public float[] inReal;
        public float[] inImag;
        public float[] outReal;
        public float[] outImag;
        public float[] t;

        public SVMBuffer inRealBuf;
        public SVMBuffer outRealBuf;
        public SVMBuffer iotaT;

        @Setup(Level.Trial)
        public void doSetup() {
            inReal = new float[size];
            t = new float[size];

            for(int i = 0; i < size; i++){
                t[i] = (float)i;
                inReal[i] = (float)Math.PI * i;
            }
            inRealBuf = SVMBuffer.fromArray(SPECIES_SVM, inReal);
        }
        @Setup(Level.Invocation)
        public void doInvocation(){
            outReal = new float[size];
            outRealBuf = SVMBuffer.fromArray(SPECIES_SVM, outReal);
            iotaT = SVMBuffer.Iota(SPECIES_SVM, size);
        }

        @TearDown(Level.Invocation)
        public void doTearDown(){
            outRealBuf.releaseSVMBuffer();
            iotaT.releaseSVMBuffer();
        }
    }

    @Benchmark
    @BenchmarkMode(Mode.AverageTime)
    public void DFTSVM(BenchmarkSetup state){
        DFT.computeSVM(state.inRealBuf, state.outReal, state.iotaT);
    }

    // @Benchmark
    // @BenchmarkMode(Mode.AverageTime)
    // public void DFTOpenCL(BenchmarkSetup state){
    //     DFT.computeOpenCL(state.inRealBuf, state.outRealBuf);
    // }

    // @Benchmark
    // @BenchmarkMode(Mode.AverageTime)
    // public void DFTOpenCLWithCopy(BenchmarkSetup state){
    //     var inRealBuf = SVMBuffer.fromArray(SPECIES_SVM, state.inReal);
    //     var outRealBuf = SVMBuffer.fromArray(SPECIES_SVM, state.outReal);

    //     DFT.computeOpenCL(inRealBuf, outRealBuf);

    //     inRealBuf.intoArray(state.inReal);
    //     outRealBuf.intoArray(state.outReal);
    // }


    // @Benchmark
    // @BenchmarkMode(Mode.AverageTime)
    // public void DFTAVX(BenchmarkSetup state){
    //     DFT.computeAVX(state.inReal, state.outReal, state.t);
    // }

    // @Benchmark
    // @BenchmarkMode(Mode.AverageTime)
    // public void KernelBuilderWithCopy(BenchmarkSetup state){
    //     DFT.computeKernelBuilder(state.inReal, state.outReal);
    // }

    // @Benchmark
    // @BenchmarkMode(Mode.AverageTime)
    // public void DFTSerial(BenchmarkSetup state){
    //     DFT.computeSerial(state.inReal, state.outReal);
    // }
}