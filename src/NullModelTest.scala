import scalation.linalgebra.{MatriD, MatrixD, VectoD, VectorD,VectorI,MatrixI}
import scalation.analytics.NullModel

object NullModelTest extends App
{
    // 4 data points:
    val y = VectorD (1, 3, 3, 4)                        // y vector

    println ("y = " + y)

    val rg = new NullModel (y)
    rg.train ().eval ()

    println ("coefficient = " + rg.coefficient)
    println ("            = " + rg.fitLabels)
    println ("fit         = " + rg.fit)

    val z  = VectorD (5)                                // predict y for one point
    val yp = rg.predict (z)
    println ("predict (" + z + ") = " + yp)

} // NullModelTest object