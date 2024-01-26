package ch.wakeLanaka;

import org.openjdk.jmh.annotations.*;
import org.openjdk.jmh.infra.Blackhole;
import jdk.incubator.vector.SVMBuffer;
import jdk.incubator.vector.GPUInformation;

import java.util.concurrent.TimeUnit;

@State(Scope.Benchmark)
public class FmaArrayBenchmark {

        private static final GPUInformation SPECIES = SVMBuffer.SPECIES_PREFERRED;

   // All these numbers are 2^n-1 to avoid memory alignment!
   @Param({"15", "255", "4095", "65535", "1048575"})
   // @Param({"16777215"})
   // @Param({"268435455"})
   private int LENGTH;
   private float[] a;
   private float[] b;
   private SVMBuffer bufferA;
   private SVMBuffer bufferB;
   private SVMBuffer bufferC;


   @Setup(Level.Iteration)
   public void init(){
        this.a = GeneratorHelpers.initFloatArray(LENGTH);
        // this.b = GeneratorHelpers.initFloatArray(LENGTH);
        this.bufferA = SVMBuffer.fromArray(SPECIES, a);
        this.bufferB = SVMBuffer.fromArray(SPECIES, a);
        this.bufferC = SVMBuffer.zero(SPECIES, this.bufferA.length, this.bufferA.type);
   }

   @TearDown(Level.Iteration)
   public void tearDown(){
        this.bufferA.releaseSVMBuffer();
        this.bufferB.releaseSVMBuffer();
        this.bufferC.releaseSVMBuffer();
   }

   @Benchmark
   public void arrayFmaScalar(Blackhole bh){
       bh.consume(FmaArray.scalarFMA(a, a));
   }

   @Benchmark
   public void arrayFmaVector(Blackhole bh){
       bh.consume(FmaArray.vectorFMA(a, a));
   }

   @Benchmark
   public void arrayFmaGPU(Blackhole bh){
       bh.consume(FmaArray.gpuFMA(bufferA, bufferB, bufferC));
   }

   @Benchmark
   public void arrayFmaGPUCopy(Blackhole bh){
       bh.consume(FmaArray.gpuFMACopy(a, a));
   }
}
