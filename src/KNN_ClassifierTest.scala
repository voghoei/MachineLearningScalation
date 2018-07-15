import scalation.linalgebra.{MatriD, MatrixD, VectoD, VectorD,VectorI}
import scalation.math.double_exp
import scalation.plot.Plot
import scalation.util.Error
import scalation.analytics.classifier.KNN_Classifier

object KNN_ClassifierTest extends App
{
    //                            x1 x2  y
    val xy = new MatrixD ((10, 3), 1, 5, 1,       // joint data matrix
                                   2, 4, 1,
                                   3, 4, 1,
                                   4, 4, 1,
                                   5, 3, 0,
                                   6, 3, 1,
                                   7, 2, 0,
                                   8, 2, 0,
                                   9, 1, 0,
                                  10, 1, 0)

    val fn = Array ("x1", "x2")                   // feature/variable names
    val cn = Array ("No", "Yes")                  // class names

    println ("----------------------------------------------------")
    println ("xy = " + xy)

    val cl = KNN_Classifier (xy, fn, 2, cn)

    val z1 = VectorD (10.0, 10.0)
    println ("----------------------------------------------------")
    println ("z1 = " + z1)
    println ("class = " + cl.classify (z1))

    val z2 = VectorD ( 3.0,  3.0)
    println ("----------------------------------------------------")
    println ("z2 = " + z2)
    println ("class = " + cl.classify (z2))

    val y  =  xy.col (xy.dim2-1).toInt
    val yp = VectorI (for (i <- xy.range1) yield cl.classify (xy(i).slice (0, xy.dim2-1))._1)
    println (cl.actualVpredicted (y, yp))

    new Plot (xy.col(0), y.toDouble, yp.toDouble)

} // KNN_ClassifierTest object