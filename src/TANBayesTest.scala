import scalation.linalgebra.{MatriD, MatrixD, VectoD, VectorD,VectorI,MatrixI}
import scalation.analytics.classifier.TANBayes
import scalation.analytics.classifier.TANBayes0

object TANBayesTest extends App
{
    // training-set -----------------------------------------------------------
    // x0: Fast
    // x1: Strong
    // y:  Classification (No/0, Yes/1)
    // features:                 x0 x1  y
    val xy = new MatrixI((10, 3), 1, 1, 1,
                                  1, 1, 1,
                                  1, 0, 1,
                                  1, 0, 1,
                                  1, 0, 0,
                                  0, 1, 0,
                                  0, 1, 0,
                                  0, 1, 1,
                                  0, 0, 0,
                                  0, 0, 0)

    val fn = Array ("Fast", "Strong")                   // feature names
    val cn = Array ("No", "Yes")                        // class names

    println("xy = " + xy)
    println("---------------------------------------------------------------")

    val tan0 = TANBayes0 (xy, fn, 2, cn, 1, null)       // create the classifier
    val tan  = TANBayes  (xy, fn, 2, cn, 1, null)        // create the classifier

    // train the classifier ---------------------------------------------------
    tan0.train ()
    tan.train ()

    // test sample ------------------------------------------------------------
    val z = VectorI (1, 0) // new data vector to classify
    println ("Use tan0 to classify (" + z + ") = " + tan0.classify (z))
    println ("Use tan  to classify (" + z + ") = " + tan.classify (z))

    println ("tan0 cv accu = " + tan0.crossValidateRand())    // cross validate the classifier
    println ("tan  cv accu = " + tan.crossValidateRand())     // cross validate the classifier

} // TANBayesTest object