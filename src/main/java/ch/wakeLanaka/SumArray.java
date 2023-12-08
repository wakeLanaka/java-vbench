package ch.wakeLanaka;

import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorSpecies;

import jdk.incubator.vector.GPUVector;
import jdk.incubator.vector.GPUSpecies;

import jdk.incubator.vector.SVMBuffer;
import jdk.incubator.vector.GPUInformation;


public class SumArray {


    private static final GPUInformation SPECIES_SVM = SVMBuffer.SPECIES_PREFERRED;

    private static final VectorSpecies<Float> SPECIES = FloatVector.SPECIES_PREFERRED;
    public static float[] scalarComputation(float[] a, float[] b) {
        var c = new float[a.length];

        for (var i = 0; i < a.length; i++) {
            c[i] = a[i] + b[i];
        }

        return c;
    }

    public static float[] vectorComputation(float[] a, float[] b) {
        var c = new float[a.length];
        var upperBound = SPECIES.loopBound(a.length);

        var i = 0;
        for (; i < upperBound; i += SPECIES.length()) {
            var va = FloatVector.fromArray(SPECIES, a, i);
            var vb = FloatVector.fromArray(SPECIES, b, i);
            var vc = va.add(vb);
            vc.intoArray(c, i);
        }

        for (; i < a.length; i++) { // Cleanup loop
            c[i] = a[i] + b[i];
        }

        return c;

    }

    // public static int[] gpuComputation(int[] a, int[] b) {
    //     var c = new int[a.length];
    //     var upperBound_gpu = SPECIES_GPU.loopBound(a.length);
    //     for(int i = 0; i < upperBound_gpu; i += SPECIES_GPU.length(i)) {
    //         var va1 = GPUVector.fromArray(SPECIES_GPU, a, i);
    //         var vb1 = GPUVector.fromArray(SPECIES_GPU, b, i);
    //         var vc1 = va1.add(vb1);
    //         vc1.intoArray(c, i);
    //     }
    //     return c;
    // }

    public static float[] gpuZeroCopyAdd(float[] a, float[] b) {
        var va = GPUVector.fromArray(SPECIES_SVM, a);
        var vb = GPUVector.fromArray(SPECIES_SVM, b);
        var vc = va.Add(vb);
        return vc.array;
    }

    public static float[] gpuSVMAddition(SVMBuffer bufferA, SVMBuffer bufferB) {
        float[] c = new float[bufferA.length];
        var bufferC = bufferA.Add(bufferB);
        bufferC.releaseSVMBuffer();
        return c;
    }

    public static float[] gpuSVMCopyAddition(float[] a, float[] b) {
        float[] c = new float[a.length];
        SVMBuffer bufferA = SVMBuffer.fromArray(SPECIES_SVM, a);
        SVMBuffer bufferB = SVMBuffer.fromArray(SPECIES_SVM, b);
        var bufferC = bufferA.Add(bufferB);
        bufferC.intoArray(c);
        return c;
    }
}
