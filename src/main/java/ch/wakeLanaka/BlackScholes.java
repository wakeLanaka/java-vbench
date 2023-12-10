package ch.wakeLanaka;

import java.lang.Math;
import jdk.incubator.vector.SVMBuffer;
import jdk.incubator.vector.GPUInformation;

import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorOperators;
import jdk.incubator.vector.VectorSpecies;
import jdk.incubator.vector.KernelBuilder;
import jdk.incubator.vector.KernelBuilder.KernelStatement;

public class BlackScholes {
    private static final GPUInformation SPECIES_SVM = SVMBuffer.SPECIES_PREFERRED;
    static final VectorSpecies<Float> fsp = FloatVector.SPECIES_PREFERRED;

    static final float Y = 0.2316419f;
    static final float A1 = 0.31938153f;
    static final float A2 = -0.356563782f;
    static final float A3 = 1.781477937f;
    static final float A4 = -1.821255978f;
    static final float A5 = 1.330274429f;
    static final float PI = (float)Math.PI;

    private static float cdf(float inp) {
        float x = inp;
        if (inp < 0f) {
            x = -inp; 
        }


        float term = 1f / (1f + (Y * x));
        float term_pow2 = term * term;
        float term_pow3 = term_pow2 * term;
        float term_pow4 = term_pow2 * term_pow2;
        float term_pow5 = term_pow2 * term_pow3;

        float part1 = (1f / (float)Math.sqrt(2f * PI)) * (float)Math.exp((-x * x) * 0.5f);

        float part2 = (A1 * term) +
                      (A2 * term_pow2) +
                      (A3 * term_pow3) +
                      (A4 * term_pow4) +
                      (A5 * term_pow5);

        if (inp >= 0f)
            return 1f - part1 * part2;
        else
            return part1 * part2;
    }

    public static void computeJava(float sig, float r, float[] x, float[] call, float[] put, float[] t, float[] s0, int size, int offset){
        float sig_sq_by2 = 0.5f * sig * sig;
        for (int i = offset; i < size; i++ ) {
            float log_s0byx = (float)Math.log(s0[i] / x[i]);
            float sig_sqrt_t = sig * (float)Math.sqrt(t[i]);
            float exp_neg_rt = (float)Math.exp(-r * t[i]);
            float d1 = (log_s0byx + (r + sig_sq_by2) * t[i])/(sig_sqrt_t);
            float d2 = d1 - sig_sqrt_t;
            if(i == 0 || i == 1023){
                call[i] = s0[i] * cdf(d1) - exp_neg_rt * x[i] * cdf(d2);
                put[i]  = call[i] + exp_neg_rt - s0[i];
            }
       }
    }

    // https://github.com/openjdk/jdk/pull/5234/commits/be7c701b031c3c616d85d79e2ea6a16c93eb6b90#diff-bf4f6a686f1d218a004427b1f8fa21285c3f4da359cbf67a6b89f4d3d07ba838R63
   private static FloatVector vcdf(FloatVector vinp) {
        var vx = vinp.abs();
        var vone = FloatVector.broadcast(fsp, 1.0f);
        var vtwo = FloatVector.broadcast(fsp, 2.0f);
        var vterm = vone.div(vone.add(vx.mul(Y)));
        var vterm_pow2 = vterm.mul(vterm);
        var vterm_pow3 = vterm_pow2.mul(vterm);
        var vterm_pow4 = vterm_pow2.mul(vterm_pow2);
        var vterm_pow5 = vterm_pow2.mul(vterm_pow3);
        var vpart1 = vone.div(vtwo.mul(PI).lanewise(VectorOperators.SQRT)).mul(vx.mul(vx).neg().mul(0.5f).lanewise(VectorOperators.EXP));
        var vpart2 = vterm.mul(A1).add(vterm_pow2.mul(A2)).add(vterm_pow3.mul(A3)).add(vterm_pow4.mul(A4)).add(vterm_pow5.mul(A5));
        var vmask = vinp.compare(VectorOperators.GT, 0f);
        var vresult1 = vpart1.mul(vpart2);
        var vresult2 = vresult1.neg().add(vone);
        var vresult = vresult1.blend(vresult2, vmask);

        return vresult;
    }

    public static int computeAVX(float sig, float r, float[] x, float[] call, float[] put, float[] t, float[] s0) {
        int i = 0;
        var vsig = FloatVector.broadcast(fsp, sig);
        var vsig_sq_by2 = vsig.mul(vsig).mul(0.5f);
        var vr = FloatVector.broadcast(fsp, r);
        var vnegr = FloatVector.broadcast(fsp, -r);
        for (; i <= x.length - fsp.length(); i += fsp.length()) {
            var vx = FloatVector.fromArray(fsp, x, i);
            var vs0 = FloatVector.fromArray(fsp, s0, i);
            var vt = FloatVector.fromArray(fsp, t, i);
            var vlog_s0byx = vs0.div(vx).lanewise(VectorOperators.LOG);
            var vsig_sqrt_t = vt.lanewise(VectorOperators.SQRT).mul(vsig);
            var vexp_neg_rt = vt.mul(vnegr).lanewise(VectorOperators.EXP);
            var vd1 = vsig_sq_by2.add(vr).mul(vt).add(vlog_s0byx).div(vsig_sqrt_t);
            var vd2 = vd1.sub(vsig_sqrt_t);
            var vcall = vs0.mul(vcdf(vd1)).sub(vx.mul(vexp_neg_rt).mul(vcdf(vd2)));
            var vput = vcall.add(vexp_neg_rt).sub(vs0);
            vcall.intoArray(call, i);
            vput.intoArray(put, i);
        }
        return i;
    }


    private static SVMBuffer svmcdf(SVMBuffer vinp) {
        float[] help = new float[vinp.length];
        var vx = vinp.Abs();
        var vone = SVMBuffer.Broadcast(SPECIES_SVM, 1.0f, vinp.length);
        var vtwo = SVMBuffer.Broadcast(SPECIES_SVM, 2.0f, vinp.length);
        var vterm = vone.Division(vone.Add(vx.Multiply(Y)));
        var vterm_pow2 = vterm.Multiply(vterm);
        var vterm_pow3 = vterm_pow2.Multiply(vterm);
        var vterm_pow4 = vterm_pow2.Multiply(vterm_pow2);
        var vterm_pow5 = vterm_pow2.Multiply(vterm_pow3);
        var vpart1 = vone.Division(vtwo.MultiplyInPlace(PI).SqrtInPlace()).MultiplyInPlace(vx.MultiplyInPlace(vx).MultiplyInPlace(-1.0f).MultiplyInPlace(0.5f).ExpInPlace());
        var vpart2 = vterm.MultiplyInPlace(A1).AddInPlace(vterm_pow2.MultiplyInPlace(A2)).AddInPlace(vterm_pow3.MultiplyInPlace(A3)).AddInPlace(vterm_pow4.MultiplyInPlace(A4)).AddInPlace(vterm_pow5.MultiplyInPlace(A5)); 
        var vmask = vinp.CompareGT(0f);
        var vresult1 = vpart1.MultiplyInPlace(vpart2);
        var vresult2 = vresult1.Multiply(-1.0f).AddInPlace(vone);
        var vresult = vresult1.BlendInPlace(vresult2, vmask);

        return vresult;
    }

    public static void computeSVM(SVMBuffer vsig, SVMBuffer vr, SVMBuffer vnegr, SVMBuffer vx, SVMBuffer vcall, SVMBuffer vput, SVMBuffer vt, SVMBuffer vs0){
        var vsig_sq_by2 = vsig.Multiply(vsig).MultiplyInPlace(0.5f);
        var vlog_s0byx = vs0.Division(vx).LogInPlace();
        var vsig_sqrt_t = vt.Sqrt().MultiplyInPlace(vsig);
        var vexp_neg_rt = vt.Multiply(vnegr).ExpInPlace();
        var vd1 = vsig_sq_by2.AddInPlace(vr).MultiplyInPlace(vt).AddInPlace(vlog_s0byx).DivisionInPlace(vsig_sqrt_t);
        var vd2 = vd1.Subtract(vsig_sqrt_t);
        vcall = vs0.Multiply(svmcdf(vd1)).SubtractInPlace(vx.MultiplyInPlace(vexp_neg_rt).MultiplyInPlace(svmcdf(vd2)));
        vput = vcall.Add(vexp_neg_rt).SubtractInPlace(vs0);
    }

    public static void computeOpenCL(float sig, float r, SVMBuffer x, SVMBuffer call, SVMBuffer put, SVMBuffer t, SVMBuffer s0){
        SVMBuffer.BlackScholes(SPECIES_SVM, sig, r, x, call, put, t, s0);
    }

    public static KernelStatement buildercdf(KernelBuilder builder, KernelStatement vinp){
        var vx1 = vinp.Abs();
        var vone = builder.Var(1.0f);
        var vtwo = builder.Var(2.0f);
        var vterm = vone.Div(vone.Add(vx1.Mul(Y)));
        var vterm_pow2 = vterm.Mul(vterm);
        var vterm_pow3 = vterm_pow2.Mul(vterm);
        var vterm_pow4 = vterm_pow2.Mul(vterm_pow2);
        var vterm_pow5 = vterm_pow2.Mul(vterm_pow3);
        var vpart1 = vone.Div(vtwo.Mul(PI).Sqrt()).Mul(vx1.Mul(vx1).Mul(-0.5f).Exp());
        var vpart2 = vterm.Mul(A1).Add(vterm_pow2.Mul(A2)).Add(vterm_pow3.Mul(A3)).Add(vterm_pow4.Mul(A4)).Add(vterm_pow5.Mul(A5));
        var vmask = vinp.CompareGT(0f);
        var vresult1 = vpart1.Mul(vpart2);
        var vresult2 = vresult1.Mul(-1).Add(1);
        return vresult1.Blend(vresult2, vmask);
    }

    public static void computeKernelBuilder(float sig, float r, SVMBuffer vx, SVMBuffer vcall, SVMBuffer vput, SVMBuffer vt, SVMBuffer vs0){

        var builder = new KernelBuilder(vx.length);

        var vsig_sq_by2 = builder.Var(sig).Mul(sig).Mul(0.5f);
        var vlog_s0byx = builder.Var(vs0).Div(vx).Log();
        var vsig_sqrt_t = builder.Var(vt).Sqrt().Mul(sig);
        var vexp_neg_rt = builder.Var(vt).Mul(-r).Exp();
        var vd1 = vlog_s0byx.Add(r).Mul(vt).Add(vlog_s0byx).Div(vsig_sqrt_t);
        var vd2 = vd1.Sub(vsig_sqrt_t);
        var vcdf1 = buildercdf(builder, vd1);
        var vcdf2 = buildercdf(builder, vd2);
        builder.Assign(vcall, builder.Var(vs0).Mul(vcdf1).Sub(builder.Var(vx).Mul(vexp_neg_rt).Mul(vcdf2)));
        builder.Assign(vput, builder.Var(vcall).Add(vexp_neg_rt).Sub(vs0));
        builder.ExecKernel(SPECIES_SVM, builder);
    }
}
