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

    public static void computeSerial(float[] inReal, float[] outReal, float[] inImag, float[] outImag){
        int n = inReal.length;
        float twoPI = 2 * (float)Math.PI;

        for(int k = 0; k < n; k++){
            float sumreal = 0;
            float sumimag = 0;
            for(int t = 0; t < n; t++){
                float angle = twoPI * t * k / n;
                sumreal += inReal[t] * (float)Math.cos(angle) + (inImag[t] * (float)Math.sin(angle));
                sumimag += -inReal[t] * (float)Math.sin(angle) + (inImag[t] * (float)Math.cos(angle));
            }
            outReal[k] = sumreal;
            outImag[k] = sumimag;
        }
    }

    public static void computeAVX(float[] inReal, float[] outReal, float[] inImag, float[] outImag, float[] t){
        int n = inReal.length;
        float twoPI = 2 * (float)Math.PI;

        for(int k = 0; k < n; k++){
            float sumReal = 0;
            float sumImag = 0;
            int i = 0;
            for (i = 0; i <= inReal.length - fsp.length(); i += fsp.length()) {
                var vt = FloatVector.fromArray(fsp, t, i);
                var vinReal = FloatVector.fromArray(fsp, inReal, i);
                var vinImag = FloatVector.fromArray(fsp, inImag, i);
                var angle = vt.mul(k).mul(twoPI).div(n);
                sumReal += angle.lanewise(VectorOperators.COS).mul(vinReal).add(vinImag.mul(angle.lanewise(VectorOperators.SIN))).reduceLanes(VectorOperators.ADD);
                sumImag += angle.lanewise(VectorOperators.SIN).mul(vinReal).neg().add(vinImag.mul(angle.lanewise(VectorOperators.COS))).reduceLanes(VectorOperators.ADD);
            }
            for(; i < n; i++){
                float angle = twoPI * i * k / n;
                sumReal += (inReal[i] * (float)Math.cos(angle)) + (inImag[i] * (float)Math.sin(angle));
                sumImag += -(inReal[i] * (float)Math.sin(angle)) + (inImag[i] * (float)Math.cos(angle));
            }
            outReal[k] = sumReal;
            outImag[k] = sumImag;
        }
    }

    public static void computeKernelBuilder(float[] inReal, float[] outReal, float[] inImag, float[] outImag){
        int n = inReal.length;

        var loop = ForKernelBuilder.For(n, 0, 1, 1);
            var vinReal = SVMBuffer.fromArray(loop.getInfo(), inReal);
            var voutReal = SVMBuffer.fromArray(loop.getInfo(), outReal);
            var vinImag = SVMBuffer.fromArray(loop.getInfo(), inImag);
            var voutImag = SVMBuffer.fromArray(loop.getInfo(), outImag);

            var angle = loop.body.Iota().Mul(2).Mul((float)Math.PI).Mul(loop.getIndex()).Div(n);
            loop.body.AddAssign(voutReal, angle.Cos().Mul(vinReal, loop.getIndex()).Add(angle.Sin().Mul(vinImag, loop.getIndex())));
            loop.body.AddAssign(voutImag, angle.Sin().Mul(vinReal, loop.getIndex()).Mul(-1).Add(angle.Cos().Mul(vinImag, loop.getIndex())));
        loop.End();

        voutReal.intoArray(outReal);
        voutImag.intoArray(outImag);
        vinReal.releaseSVMBuffer();
        voutReal.releaseSVMBuffer();
        vinImag.releaseSVMBuffer();
        voutImag.releaseSVMBuffer();
    }

    public static void computeSVM(SVMBuffer inReal, float[] outReal, SVMBuffer inImag, float[] outImag, SVMBuffer iotaT){
        int n = inReal.length;
        float twoPI = 2 * (float)Math.PI;

        for(int k = 0; k < n; k++) {
            var angle = iotaT.mul(twoPI).mulInPlace(k).divInPlace(n);
            var real1 = angle.cos().mulInPlace(inReal);
            var imag1 = angle.sin().mulInPlace(inImag);
            outReal[k] = real1.addInPlace(imag1).sumReduceFloat();
            var real2 = angle.sin().mulInPlace(inReal).mulInPlace(-1);
            var imag2 = angle.cos().mulInPlace(inImag);
            outImag[k] = real2.addInPlace(imag2).sumReduceFloat();
            angle.releaseSVMBuffer();
            real1.releaseSVMBuffer();
            imag1.releaseSVMBuffer();
            real2.releaseSVMBuffer();
            imag2.releaseSVMBuffer();
        }
    }
}
