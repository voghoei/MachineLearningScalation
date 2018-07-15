import scalation.linalgebra.{MatriD, MatrixD, VectoD, VectorD,VectorI}
import scalation.analytics.classifier.LDA
import scalation.util.banner
import scalation.plot.Plot

object LDATest extends App
{
    // features/variable: 
    // x1: curvature
    // x2: diameter
    //                           x1    x2
    val x = new MatrixD ((7, 2), 2.95, 6.63,
                                 2.53, 7.79,
                                 3.57, 5.65,
                                 3.16, 5.47,
                                 2.58, 4.46,
                                 2.16, 6.22,
                                 3.27, 3.52)
    val y = VectorI (0, 0, 0, 0, 1, 1, 1)

    val fn = Array ("curvature", "diameter")                   // feature names
    val cn = Array ("pass", "fail")                            // class names
    val cl = new LDA (x, y, fn, 2, cn)                         // create the LDA classifier
    cl.train ()

    banner ("classify")
    val z  = VectorD (2.81, 5.46)
    println (s"classify ($z) = ${cl.classify (z)}")

    banner ("test")
    val yp = new VectorI (x.dim1)                              // predicted class vector
    for (i <- x.range1) yp(i) = cl.classify (x(i))._1
    println (s" y = $y \n yp = $yp")
    println (cl.actualVpredicted (y, yp))                      // compare y vs. yp

    val t = VectorD.range (0, x.dim1)
    new Plot (t, y.toDouble, yp.toDouble, "y(black)/yp(red) vs. t")
    new Plot (x.col(0), y.toDouble, yp.toDouble, "y(black)/yp(red) vs. x1")
    new Plot (x.col(1), y.toDouble, yp.toDouble, "y(black)/yp(red) vs. x2")

} // LDATestObject