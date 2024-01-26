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

    public static SVMBuffer gpuSVMAddition(SVMBuffer bufferA, SVMBuffer bufferB) {
        var bufferC = bufferA.add(bufferB);
        return bufferC;
    }

    public static float[] gpuSVMCopyAddition(float[] a, float[] b) {
        float[] c = new float[a.length];
        SVMBuffer bufferA = SVMBuffer.fromArray(SPECIES_SVM, a);
        SVMBuffer bufferB = SVMBuffer.fromArray(SPECIES_SVM, b);
        var bufferC = bufferA.add(bufferB);
        bufferC.intoArray(c);
        bufferA.releaseSVMBuffer();
        bufferB.releaseSVMBuffer();
        bufferC.releaseSVMBuffer();
        return c;
    }
}
