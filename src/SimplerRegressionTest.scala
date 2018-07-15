import scalation.linalgebra.{MatriD, MatrixD, VectoD, VectorD}
import scalation.math.double_exp
import scalation.plot.Plot
import scalation.util.Error
import scalation.analytics.SimplerRegression

object SimplerRegressionTest extends App
{
    // 4 data points:
    val x = VectorD (1, 2, 3, 4)
    val y = VectorD (1, 3, 3, 4)
//  val y = VectorD (1, 3, 2, 4)

    println ("x = " + x)
    println ("y = " + y)

    val rg = SimplerRegression (x, y)
    rg.train ().eval ()

    println ("coefficient = " + rg.coefficient)
    println ("            = " + rg.fitLabels)
    println ("fit         = " + rg.fit)

}
