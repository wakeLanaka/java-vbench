package ch.wakeLanaka;

import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.IntVector;
import jdk.incubator.vector.VectorOperators;
import jdk.incubator.vector.VectorSpecies;

import jdk.incubator.vector.SVMBuffer;
import jdk.incubator.vector.GPUInformation;

public class FmaArray {

    private static final VectorSpecies<Float> SPECIES = FloatVector.SPECIES_PREFERRED;
    private static final GPUInformation SPECIES_SVM = SVMBuffer.SPECIES_PREFERRED;

    // FMA: Fused Multiply Add: c = c + (a * b)
    public static float[] scalarFMA(float[] a, float[] b){
        var c = new float[a.length];

        for(var i=0; i < a.length; i++){
            c[i] = Math.fma(a[i], b[i], c[i]);
        }
        return c;
    }

    public static SVMBuffer gpuFMA(SVMBuffer bufferA, SVMBuffer bufferB, SVMBuffer bufferC){
        var res = bufferA.fma(bufferB, bufferC);

        return res;
    }

    public static float[] gpuFMACopy(float[] a, float[] b){
        float[] c = new float[a.length];

        var bufferA = SVMBuffer.fromArray(SPECIES_SVM, a);
        var bufferB = SVMBuffer.fromArray(SPECIES_SVM, b);
        var bufferC = SVMBuffer.fromArray(SPECIES_SVM, c);

        var res = bufferA.fma(bufferB, bufferC);

        res.intoArray(c);

        res.releaseSVMBuffer();
        bufferA.releaseSVMBuffer();
        bufferB.releaseSVMBuffer();
        bufferC.releaseSVMBuffer();

        return c;
    }

    public static float[] vectorFMA(float[] a, float[] b){
        var upperBound = SPECIES.loopBound(a.length);
        var c = new float[a.length];

        var i = 0;
        for (; i < upperBound; i += SPECIES.length()) {
            // FloatVector va, vb, vc
            var sum = FloatVector.zero(SPECIES);
            var va = FloatVector.fromArray(SPECIES, a, i);
            var vb = FloatVector.fromArray(SPECIES, b, i);
            sum = va.fma(vb, sum);
            sum.intoArray(c, i);
        }

        for (; i < a.length; i++) {
            c[i] = Math.fma(a[i], b[i], c[i]);
        }
        return c;
    }

}
