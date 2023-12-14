package ch.wakeLanaka;

import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorSpecies;
import jdk.incubator.vector.GPUInformation;
import jdk.incubator.vector.SVMBuffer;

public class MatrixMul {
    private static final VectorSpecies<Float> SPECIES = FloatVector.SPECIES_PREFERRED;

    public static float[] computeSerial(float[] a, float[] b, int n) {
        float[] c = new float[n * n];

        for (int i = 0; i < n; i++) {
            for (int k = 0; k < n; k++) {
                float aik = a[i * n + k];
                for (int j = 0; j < n; j++) {
                    c[i * n + j] = Math.fma(aik, b[k * n + j], c[i * n + j]);
                }
            }
        }
        return c;
    }

    public static float[] computeAVX(float[] a, float[] b, int n) {
        final int upperBound = SPECIES.loopBound(n);
        float[] c = new float[n * n];

        for (int i = 0; i < n; i++) {
            for (int k = 0; k < n; k++) {
                float aik = a[i * n + k];
                FloatVector vaik = FloatVector.broadcast(SPECIES, aik);
                for (int j = 0; j < upperBound; j += SPECIES.length()) {
                    FloatVector vb = FloatVector.fromArray(SPECIES, b, k * n + j);
                    FloatVector vc = FloatVector.fromArray(SPECIES, c, i * n + j);
                    vc = vaik.fma(vb, vc);
                    vc.intoArray(c, i * n + j);
                }
            }
        }
        return c;
    }

    public static SVMBuffer computeSVM(SVMBuffer a, SVMBuffer b, SVMBuffer c, int n) {
        for (int i = 0; i < n; i++) {
            c = a.matrixFma(b, c, n, n, i);
        }
        return c;
    }
}
