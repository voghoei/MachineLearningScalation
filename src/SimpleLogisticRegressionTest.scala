import scalation.linalgebra.{MatriD, MatrixD, VectoD, VectorD,VectorI}
import scalation.math.double_exp
import scalation.plot.Plot
import scalation.util.Error
import scalation.analytics.classifier.SimpleLogisticRegression
import scalation.util.banner
import scalation.analytics.classifier.Classifier //def crossValidate (nx: Int = 10, show: Boolean = false): Double =
                                                 // def crossValidateRand (nx: Int = 10, show: Boolean = false): Double =  

import scalation.analytics.classifier.ClassifierReal //featureSelection


object SimpleLogisticRegressionTest extends App
{
    // 32 data points:            One    Mpg
    val x = new MatrixD ((32, 2), 1.0,  21.0,        //  1 - Mazda RX4 
                                  1.0,  21.0,        //  2 - Mazda RX4 Wa
                                  1.0,  22.8,        //  3 - Datsun 710
                                  1.0,  21.4,        //  4 - Hornet 4 Drive
                                  1.0,  18.7,        //  5 - Hornet Sportabout
                                  1.0,  18.1,        //  6 - Valiant
                                  1.0,  14.3,        //  7 - Duster 360
                                  1.0,  24.4,        //  8 - Merc 240D 
                                  1.0,  22.8,        //  9 - Merc 230
                                  1.0,  19.2,        // 10 - Merc 280
                                  1.0,  17.8,        // 11 - Merc 280C
                                  1.0,  16.4,        // 12 - Merc 450S
                                  1.0,  17.3,        // 13 - Merc 450SL
                                  1.0,  15.2,        // 14 - Merc 450SLC
                                  1.0,  10.4,        // 15 - Cadillac Fleetwood
                                  1.0,  10.4,        // 16 - Lincoln Continental
                                  1.0,  14.7,        // 17 - Chrysler Imperial
                                  1.0,  32.4,        // 18 - Fiat 128
                                  1.0,  30.4,        // 19 - Honda Civic
                                  1.0,  33.9,        // 20 - Toyota Corolla
                                  1.0,  21.5,        // 21 - Toyota Corona
                                  1.0,  15.5,        // 22 - Dodge Challenger
                                  1.0,  15.2,        // 23 - AMC Javelin
                                  1.0,  13.3,        // 24 - Camaro Z28
                                  1.0,  19.2,        // 25 - Pontiac Firebird
                                  1.0,  27.3,        // 26 - Fiat X1-9
                                  1.0,  26.0,        // 27 - Porsche 914-2
                                  1.0,  30.4,        // 28 - Lotus Europa
                                  1.0,  15.8,        // 29 - Ford Pantera L
                                  1.0,  19.7,        // 30 - Ferrari Dino
                                  1.0,  15.0,        // 31 - Maserati Bora
                                  1.0,  21.4)        // 32 - Volvo 142E

    // V/S (e.g., V-6 vs. I-4)
    val y = VectorI (0, 0, 1, 1, 0, 1, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0,
                     0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1)

    var z: VectoD = null

    println ("x = " + x)
    println ("y = " + y)

    val fn = Array ("One", "Mpg")

    val rg = new SimpleLogisticRegression (x, y, fn)
    rg.train_null ()                                    // train based on null model
    rg.train ()                                         // train based on full model

    banner ("Simple Logistic Regression Results")
    println ("b = " + rg.coefficient)
    println (rg.fitLabels)
    println (rg.fit)

    z = VectorD (1.0, 15.0)                             // classify point z
    println ("classify (" + z + ") = " + rg.classify (z))

    z = VectorD (1.0, 30.0)                             // classify point z
    println ("classify (" + z + ") = " + rg.classify (z))

    println ("acc = " + rg.crossValidate (10, true))

    for (b_0 <- -9.0 to -8.0 by 0.1; b_1 <- 0.0 to 1.0 by 0.1) {
        val b = VectorD (b_0, b_1)
        println (s"ll ($b) = ${rg.ll (b)}")
    } // for

    val yp = VectorD (for (i <- x.range1) yield rg.classify (x(i))._1.toDouble)

    new Plot (x.col(1), y.toDouble, yp)

} // SimpleLogisticRegressionTest object
