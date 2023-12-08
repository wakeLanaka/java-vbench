package ch.wakeLanaka;

import java.lang.Math;
import jdk.incubator.vector.SVMBuffer;
import jdk.incubator.vector.KernelBuilder;
import jdk.incubator.vector.GPUInformation;

import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorOperators;
import jdk.incubator.vector.VectorSpecies;
import jdk.incubator.vector.ForKernelBuilder;

public class DFT {
    private static final GPUInformation SPECIES_SVM = SVMBuffer.SPECIES_PREFERRED;
    static final VectorSpecies<Float> fsp = FloatVector.SPECIES_PREFERRED;

    public static void computeSerial(float[] inReal, float[] outReal){
        int n = inReal.length;
        float twoPI = 2 * (float)Math.PI;

        for(int k = 0; k < n; k++){
            float sumreal = 0;

            for(int t = 0; t < n; t++){
                float angle = twoPI * t * k / n;
                sumreal += (inReal[t] * (float)Math.cos(angle));
            }

            outReal[k] = sumreal;
        }
    }

    public static void computeAVX(float[] inReal,float[] outReal, float[] t){
        int n = inReal.length;
        float twoPI = 2 * (float)Math.PI;

        for(int k = 0; k < n; k++){
            float sum = 0;
            for (int i = 0; i <= inReal.length - fsp.length(); i += fsp.length()) {
                var vt = FloatVector.fromArray(fsp, t, i);
                var angle = vt.mul(k).mul(twoPI).div(n);
                var vinReal = FloatVector.fromArray(fsp, inReal, i);
                sum += angle.lanewise(VectorOperators.COS).mul(vinReal).reduceLanes(VectorOperators.ADD);
            }
            outReal[k] = sum;
        }
    }

    public static void computeOpenCL(SVMBuffer a, SVMBuffer b){
        SVMBuffer.DFT(SPECIES_SVM, a, b);
    }

    public static void computeKernelBuilder(float[] a, float[] b){
        int n = a.length;

        var loop = ForKernelBuilder.For(0, n, 1);
            var inReal = SVMBuffer.fromArray(loop.getInfo(), a);
            var outReal = SVMBuffer.fromArray(loop.getInfo(), b);
            var angle = loop.body.Iota().Mul(2).Mul((float)Math.PI).Mul(loop.getIndex()).Div(n);
            loop.body = loop.body.AddAssign(outReal, angle.Cos().Mul(inReal, loop.getIndex()));
        loop.End();

        outReal.intoArray(b);
    }

    public static void computeSVM(SVMBuffer inReal, float[] outReal, SVMBuffer iotaT){
        int n = inReal.length;
        float twoPI = 2 * (float)Math.PI;

        for(int k = 0; k < n; k++) {
            outReal[k] = iotaT.Multiply(twoPI).MultiplyInPlace(k).DivisionInPlace(n).Cos().MultiplyInPlace(inReal).reduceAdd();
        }
    }
}
