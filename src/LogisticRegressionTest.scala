import scalation.linalgebra.{MatriD, MatrixD, VectoD, VectorD,VectorI}
import scalation.math.double_exp
import scalation.plot.Plot
import scalation.util.Error
import scalation.analytics.classifier.LogisticRegression
import scalation.util.banner

object LogisticRegressionTest extends App
{
    // 40 data points:            One     Low  Medium    High
    val x = new MatrixD ((40, 4), 1.0,  102.0,   89.0,    0.0,
                                  1.0,    7.0,  233.0,    1.0,
                                  1.0,    0.0,    4.0,   41.0,
                                  1.0,    8.0,   37.0,   13.0,
                                  1.0,   40.0,   79.0,   26.0,
                                  1.0,    0.0,  625.0,  156.0,
                                  1.0,    0.0,   12.0,   79.0,
                                  1.0,    0.0,    3.0,  119.0,
                                  1.0,  115.0,  136.0,   65.0,
                                  1.0,  428.0,  416.0,  435.0,
                                  1.0,   34.0,  174.0,   56.0,
                                  1.0,    0.0,    0.0,   37.0,
                                  1.0,   97.0,  162.0,   89.0,
                                  1.0,   56.0,   47.0,  132.0,
                                  1.0, 1214.0, 1515.0,  324.0,
                                  1.0,   30.0,  103.0,  161.0,
                                  1.0,    8.0,   11.0,  158.0,
                                  1.0,   52.0,  155.0,  144.0,
                                  1.0,  142.0,  119.0,   24.0,
                                  1.0, 1370.0, 2968.0, 1083.0,
                                  1.0,  790.0,  161.0,  231.0,
                                  1.0, 1142.0,  157.0,  131.0,
                                  1.0,    0.0,    2.0,   49.0,
                                  1.0,    0.0,    0.0,   50.0,
                                  1.0,    5.0,   68.0,   49.0,
                                  1.0,    0.0,    0.0,   48.0,
                                  1.0,    0.0,    6.0,   40.0,
                                  1.0,    1.0,    8.0,   64.0,
                                  1.0,    0.0,  998.0,  551.0,
                                  1.0,  253.0,   99.0,   60.0,
                                  1.0, 1395.0,  799.0,  244.0,
                                  1.0,    0.0,    0.0,   50.0,
                                  1.0,    1.0,   68.0,  145.0,
                                  1.0, 1318.0, 1724.0,  331.0,
                                  1.0,    0.0,    0.0,   79.0,
                                  1.0,    3.0,   31.0,   37.0,
                                  1.0,  195.0,  108.0,  206.0,
                                  1.0,    0.0,   15.0,  121.0,
                                  1.0,    0.0,  278.0,  513.0,
                                  1.0,    0.0,    0.0,  253.0)

    val y = VectorI (0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1,
                     1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 1)

    val fn = Array ("One", "Low", "Medium", "High")
    val cn = Array ("no", "yes")

    println ("x = " + x)
    println ("y = " + y)

//  val rg = new LogisticRegression (x(0 until x.dim1, 0 until 2), y, fn, cn)
    val rg = new LogisticRegression (x, y, fn, cn)
    rg.train_null ()                                    // train based on null model
    rg.train ()                                         // train based on full model

    banner ("Logistic Regression Results")
    println ("b = " + rg.coefficient)
    println (rg.fitLabels)
    println (rg.fit)

    val z  = VectorD (1.0, 100.0, 100.0, 100.0)         // classify point z
    println ("classify (" + z + ") = " + rg.classify (z))

//  new Plot (x.col(1), y, yyp)
//  new Plot (x.col(2), y, yyp)

} // LogisticRegressionTest object