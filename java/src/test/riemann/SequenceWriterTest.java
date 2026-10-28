package riemann;

import static org.junit.Assert.*;

import java.io.BufferedInputStream;
import java.io.BufferedReader;
import java.io.DataInputStream;
import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.math.BigDecimal;
import java.text.NumberFormat;

import org.junit.Test;

import math.GSeries;

public class SequenceWriterTest {
    static NumberFormat nf = NumberFormat.getInstance();
    static {
        nf.setMinimumFractionDigits(7);
        nf.setMaximumFractionDigits(7);
        nf.setGroupingUsed(false);
    }


	@Test
	public void testGramPoints() throws IOException {
        System.out.println("============= " );
		System.out.println("testGramPoints");
        System.out.println("============= " );
		double xa = 11425.704608;//243.53396107975615984;
		
		int steps = 10;
        double incr = 0.2435990469039924426798/2;
        long offset = (long) 1.0E12;
        
        for (int i = 0; i < steps; i++) {
            double x = xa + i*incr;
            double zetaFromRiemann = Riemann.riemann(x, offset);
            System.out.println(nf.format(x) +
                " "  +
                " zetaFromRiemann " + nf.format(zetaFromRiemann)
            );
        }
	}
	
	@Test
	public void testPoly7ZeroFit() throws IOException {
        System.out.println("============= " );
		System.out.println("testPoly7ZeroFit");
        System.out.println("============= " );
   	    BufferedReader[] in = Rosser.zerosFileAll("data/zerosE12.csv");
	    Poly7.epsilon *= 0.0001;
	    Poly7.derepsilon *= 0.0001;
	    Poly7.positionEpsilon *= 0.0001;
   	    int skipCount = 11675; //10, 420042;
		for (int i = 0; i < skipCount; i++) {
			CopyZeroInformation.readAndSkip(in);
		} 
   	    for (int i = 0; i < 3; i++) {
   		    CopyZeroInformation.readAndUpdateZero(in);
		} 
		double[] row = ZerosBuffer.getRow(0);
        double trans = Math.floor(row[0]);
		Poly7 poly7 = SequenceWriter.getPoly7(trans);
		double xa = poly7.a + trans;
		double xb = poly7.c + trans;
		int steps = 10;
        double incr = (xb-xa)/(steps-1);
        long offset = (long) 1.0E12;
        
        for (int i = 0; i < steps; i++) {
            double x = xa + i*incr;
            double zetaFromRiemann = Riemann.riemann(x, offset);
            System.out.println(nf.format(x) +
                " " + nf.format(poly7.eval(x-trans)) +
                " zetaFromRiemann " + nf.format(zetaFromRiemann)
            );
        }
        double[] currentMax = new double[] {0, 0};
        double[] positionMax = new double[] {0, 0};
        positionMax[0] = poly7.getPositionMax( );
        positionMax[1] = poly7.getPositionMax2(  );
        currentMax[0] = poly7.eval( positionMax[0] );
        currentMax[1] = poly7.eval(  positionMax[1]  );
        for (int i = 0; i < positionMax.length; i++) {
        	
            System.out.println("eval " + currentMax[i] +
            		" positionmax " + positionMax[i] + " " 
            		+ poly7.der(positionMax[i]));
        }
        for (int i = 0; i < positionMax.length; i++) {
			double d = positionMax[i];
            double zetaFromRiemann = Riemann.riemann(d+trans, offset);
            System.out.println(nf.format(d) +
                " " + nf.format(poly7.eval(d)) +
                " zetaFromRiemann " + nf.format(zetaFromRiemann)
            );
			
		}
        double deviation = (Math.abs(poly7.m0 -currentMax[0])
        		+ Math.abs(poly7.m1 -currentMax[1]));
        System.out.println("deviation " + deviation + " poly " + poly7);

	}
	
	@Test
	public void testReadGSeries() throws IOException {
        System.out.println("============= " );
		System.out.println("testReadGSeries");
        System.out.println("============= " );
    //double t0 = gramE12[sampleIndex][0];
        int index = 102555;
        BigDecimal offset = BigDecimal.valueOf(1.0E12);
        int k0 = 1, k1=398942;
        File file = new File("data/" + Integer.toString(index) +".dat");
        InputStream is = new FileInputStream(file);
        // create buffered input stream.
        BufferedInputStream bis = new BufferedInputStream(is);

        // create data input stream to read data in form of primitives.
        DataInputStream in = new DataInputStream(bis);
        final int initialPadding = 40;
        int R = 30000+2*initialPadding;
        double begin = in.readDouble();
        double incr = in.readDouble();
        double[][] gBeta = new double[R][2];
        for (int i = 0; i < gBeta.length; i++) {
            gBeta[i][0] = in.readDouble();
            gBeta[i][1] = in.readDouble();
        }
        GSeries gAtBeta = new GSeries(k0, k1, offset,  begin,  incr, gBeta);
        in.close();
        // line 420043
		/*
		0.76081397697, 0.95608967196, 1.06733986709, 
		4.912105763488216, -4.282347641620942, 6.646666435897174, 
		0.29893115306472573, -0.16030690662494185
t, 102565.85675401744 zeta0 0.298930479449594
t, 102566.01731244495 zeta0 -0.1603069498556976
		*/
        double[] t0 = {
        		 102565.76081397697, //102565.8567540174464339,
        		 102565.8566226,
        		 102565.85662256,
        		102565.95608967196, //102566.0173124449511088,
        		102566.017302155, 
        		102566.01730215, 
        		102566.06733986709
        		};
        double[] expected = {
        	 0, //0.29893115306472573,
        	 0.29893115306472573,
        	 0.29893115306472573,
        	0, //-0.16030690662494185, 
        	-0.16030690662494185, 
        	-0.16030690662494185, 
        	0
        	
        };
        int tIndex = 0;
        int steps = 10;
        for (double t : t0) {
            double[] gFromBLFI = gAtBeta.diagnosticBLFISumWithOffset( 
            		t , 4, initialPadding, 1.6E-10, false);
            double zeta0 = gAtBeta.riemannZeta(gFromBLFI, t);
            double der = gAtBeta.evaluateDer(t, initialPadding);
            System.out.println(tIndex + " " + nf.format(t) 
            + " gseries " + zeta0 +
            " der " + nf.format(der)
);
            assertEquals("not equal", expected[tIndex] , zeta0, 3.0E-7);
		    tIndex++;
        }
        double zerooffset = 102565;
		double xa = 0.76081397697 + zerooffset ;
		double xb = 1.0673399 + zerooffset ;
        double tabincr = (xb-xa)/(steps-1);
        
        System.out.println("============= " );
        /*
        long longOffset = (long) 1.0E12;
        for (int i = 0; i < steps; i++) {
            double x = xa + i*tabincr;
            double z = gAtBeta.evaluateZeta(x, initialPadding);
			System.out.println(nf.format(x) +
                " " + nf.format(z) +
                " der " + nf.format(gAtBeta.evaluateDer(x, initialPadding))
            );
            double zetaFromRiemann = Riemann.riemann(x, longOffset);
            System.out.println("    **  zetaFromRiemann " + 
              zetaFromRiemann + " gseries " + z);
            assertEquals("not equal", zetaFromRiemann , z, 3.0E-7);
        }
        */

    }

}
