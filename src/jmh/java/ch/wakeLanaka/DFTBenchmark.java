package ch.wakeLanaka;

import org.openjdk.jmh.annotations.*;
import java.util.Random;

import jdk.incubator.vector.SVMBuffer;
import jdk.incubator.vector.GPUInformation;

public class DFTBenchmark {
    private static final GPUInformation SPECIES_SVM = SVMBuffer.SPECIES_PREFERRED;

    @State(Scope.Thread)
    public static class BenchmarkSetup{

        @Param({"1024", "16384", "32768"})
        public int size;
        public float[] inReal;
        public float[] inImag;
        public float[] outReal;
        public float[] outImag;
        public float[] t;

        public SVMBuffer inRealBuf;
        public SVMBuffer outRealBuf;
        public SVMBuffer inImagBuf;
        public SVMBuffer outImagBuf;
        public SVMBuffer iotaT;

        @Setup(Level.Trial)
        public void doSetup() {
            inReal = new float[size];
            inImag = new float[size];
            t = new float[size];

            for(int i = 0; i < size; i++){
                t[i] = (float)i;
                inReal[i] = (float)Math.PI * i;
                inImag[i] = (float)Math.PI * i;
            }
            inRealBuf = SVMBuffer.fromArray(SPECIES_SVM, inReal);
            inImagBuf = SVMBuffer.fromArray(SPECIES_SVM, inImag);
        }
        @Setup(Level.Invocation)
        public void doInvocation(){
            outReal = new float[size];
            outImag = new float[size];
            outRealBuf = SVMBuffer.fromArray(SPECIES_SVM, outReal);
            outImagBuf = SVMBuffer.fromArray(SPECIES_SVM, outImag);
            iotaT = SVMBuffer.iota(SPECIES_SVM, size);
        }

        @TearDown(Level.Invocation)
        public void doTearDown(){
            outRealBuf.releaseSVMBuffer();
            outImagBuf.releaseSVMBuffer();
            iotaT.releaseSVMBuffer();
        }
    }

    // @Benchmark
    // @BenchmarkMode(Mode.AverageTime)
    // public void DFTTEST(BenchmarkSetup state){
    //     DFT.computeOpenCL(state.inRealBuf, state.outRealBuf, state.inImagBuf, state.outImagBuf);
    //     state.outRealBuf.intoArray(state.outReal);
    //     state.outImagBuf.intoArray(state.outImag);
    //     System.out.println("\nOpenCL REAL");
    //     System.out.println(state.outReal[0]);
    //     System.out.println(state.outReal[1]);
    //     System.out.println(state.outReal[state.size - 2]);
    //     System.out.println(state.outReal[state.size - 1]);
    //     System.out.println("OpenCL IMAG");
    //     System.out.println(state.outImag[0]);
    //     System.out.println(state.outImag[1]);
    //     System.out.println(state.outImag[state.size - 2]);
    //     System.out.println(state.outImag[state.size - 1]);

    //     state.outReal = new float[state.size];
    //     state.outImag = new float[state.size];

    //     DFT.computeSerial(state.inReal, state.outReal, state.inImag, state.outImag);
    //     System.out.println("\nSerial REAL");
    //     System.out.println(state.outReal[0]);
    //     System.out.println(state.outReal[1]);
    //     System.out.println(state.outReal[state.size - 2]);
    //     System.out.println(state.outReal[state.size - 1]);
    //     System.out.println("Serial IMAG");
    //     System.out.println(state.outImag[0]);
    //     System.out.println(state.outImag[1]);
    //     System.out.println(state.outImag[state.size - 2]);
    //     System.out.println(state.outImag[state.size - 1]);

    //     // state.outReal = new float[state.size];
    //     // state.outImag = new float[state.size];
    //     // DFT.computeSVM(state.inRealBuf, state.outReal, state.inImagBuf, state.outImag, state.iotaT);
    //     // System.out.println("\nSVM REAL");
    //     // System.out.println(state.outReal[0]);
    //     // System.out.println(state.outReal[1]);
    //     // System.out.println(state.outReal[state.size - 2]);
    //     // System.out.println(state.outReal[state.size - 1]);
    //     // System.out.println("SVM IMAG");
    //     // System.out.println(state.outImag[0]);
    //     // System.out.println(state.outImag[1]);
    //     // System.out.println(state.outImag[state.size - 2]);
    //     // System.out.println(state.outImag[state.size - 1]);

    //     // state.outReal = new float[state.size];
    //     // state.outImag = new float[state.size];

    //     // DFT.computeAVX(state.inReal, state.outReal, state.inImag, state.outImag, state.t);
    //     // System.out.println("\nAVX REAL");
    //     // System.out.println(state.outReal[0]);
    //     // System.out.println(state.outReal[1]);
    //     // System.out.println(state.outReal[state.size - 2]);
    //     // System.out.println(state.outReal[state.size - 1]);
    //     // System.out.println("AVX IMAG");
    //     // System.out.println(state.outImag[0]);
    //     // System.out.println(state.outImag[1]);
    //     // System.out.println(state.outImag[state.size - 2]);
    //     // System.out.println(state.outImag[state.size - 1]);

    //     // state.outReal = new float[state.size];
    //     // state.outImag = new float[state.size];
    //     // DFT.computeKernelBuilder(state.inReal, state.outReal, state.inImag, state.outImag);
    //     // System.out.println("\nKernelBuilder REAL");
    //     // System.out.println(state.outReal[0]);
    //     // System.out.println(state.outReal[1]);
    //     // System.out.println(state.outReal[state.size - 2]);
    //     // System.out.println(state.outReal[state.size - 1]);
    //     // System.out.println("KernelBuilder IMAG");
    //     // System.out.println(state.outImag[0]);
    //     // System.out.println(state.outImag[1]);
    //     // System.out.println(state.outImag[state.size - 2]);
    //     // System.out.println(state.outImag[state.size - 1]);
    // }

    @Benchmark
    @BenchmarkMode(Mode.AverageTime)
    public void DFTSVM(BenchmarkSetup state){
        DFT.computeSVM(state.inRealBuf, state.outReal, state.outRealBuf, state.outImag, state.iotaT);
    }

    @Benchmark
    @BenchmarkMode(Mode.AverageTime)
    public void DFTOpenCL(BenchmarkSetup state){
        DFT.computeOpenCL(state.inRealBuf, state.outRealBuf, state.inImagBuf, state.outImagBuf);
    }

    @Benchmark
    @BenchmarkMode(Mode.AverageTime)
    public void DFTOpenCLWithCopy(BenchmarkSetup state){
        var inRealBuf = SVMBuffer.fromArray(SPECIES_SVM, state.inReal);
        var outRealBuf = SVMBuffer.fromArray(SPECIES_SVM, state.outReal);
        var inImagBuf = SVMBuffer.fromArray(SPECIES_SVM, state.inImag);
        var outImagBuf = SVMBuffer.fromArray(SPECIES_SVM, state.outImag);

        DFT.computeOpenCL(inRealBuf, outRealBuf, inImagBuf, outImagBuf);

        outRealBuf.intoArray(state.outReal);
        outImagBuf.intoArray(state.outImag);

        inRealBuf.releaseSVMBuffer();
        outRealBuf.releaseSVMBuffer();
        inImagBuf.releaseSVMBuffer();
        outImagBuf.releaseSVMBuffer();
    }


    @Benchmark
    @BenchmarkMode(Mode.AverageTime)
    public void DFTAVX(BenchmarkSetup state){
        DFT.computeAVX(state.inReal, state.outReal, state.inImag, state.outImag, state.t);
    }

    @Benchmark
    @BenchmarkMode(Mode.AverageTime)
    public void KernelBuilderWithCopy(BenchmarkSetup state){
        DFT.computeKernelBuilder(state.inReal, state.outReal, state.inImag, state.outImag);
    }

    @Benchmark
    @BenchmarkMode(Mode.AverageTime)
    public void DFTSerial(BenchmarkSetup state){
        DFT.computeSerial(state.inReal, state.outReal, state.inImag, state.outImag);
    }
}
