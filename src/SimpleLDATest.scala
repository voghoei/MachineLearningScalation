import scalation.linalgebra.{MatriD, MatrixD, VectoD, VectorD,VectorI}
import scalation.analytics.classifier.SimpleLDA
import scalation.util.banner
import scalation.plot.Plot

object SimpleLDATest extends App
{
    // features/variable: 
    // x1: curvature
    //                 x1
    val x = VectorD (2.95, 2.53, 3.57, 3.16, 2.58, 2.16, 3.27)
    val y = VectorI (   0,    0,    0,    0,    1,    1,    1)

    val k  = 2                                                 // number of classes
    val fn = Array ("curvature")                               // feature name
    val cn = Array ("pass", "fail")                            // class names
    val cl = new SimpleLDA (x, y, fn, k, cn)                   // create SimpleLDA classifier
    cl.train ()

    banner ("classify")
    val z  = VectorD (2.81)
    println (s"classify ($z) = ${cl.classify (z)}")

    banner ("test")
    val yp = new VectorI (x.dim)                               // predicted class vector
    for (i <- x.range) yp(i) = cl.classify (VectorD (x(i)))._1
    println (s" y = $y \n yp = $yp")
    println (cl.actualVpredicted (y, yp))                      // compare y vs. yp

    val t = VectorD.range (0, x.dim)
    new Plot (t, y.toDouble, yp.toDouble, "y(black)/yp(red) vs. t")
    new Plot (x, y.toDouble, yp.toDouble, "y(black)/yp(red) vs. x")

} // SimpleLDATestObject