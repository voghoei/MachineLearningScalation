import scalation.linalgebra.{MatriD, MatrixD, VectoD, VectorD,VectorI,MatrixI}
import scalation.math.double_exp
import scalation.plot.Plot
import scalation.util.Error
import scalation.analytics.classifier.BayesClassifier
import BayesClassifier.{me_default, test}

object BayesClassifierTest extends App
{
    // x0: Color:   Red (1), Yellow (0)
    // x1: Type:    SUV (1), Sports (0)
    // x2: Origin:  Domestic (1), Imported (0)
    // y:  Classification (No/0, Yes/1)
    // features:                  x0 x1 x2  y
    val xy = new MatrixI ((10, 4), 1, 0, 1, 1,                 // data matrix
                                   1, 0, 1, 0,
                                   1, 0, 1, 1,
                                   0, 0, 1, 0,
                                   0, 0, 0, 1,
                                   0, 1, 0, 0,
                                   0, 1, 0, 1,
                                   0, 1, 1, 0,
                                   1, 1, 0, 0,
                                   1, 0, 0, 1)

    val fn = Array ("Color", "Type", "Origin")                 // feature/variable names
    val k  = 2                                                 // number of classes
    val cn = Array ("No", "Yes")                               // class names
    val vc = null.asInstanceOf [Array [Int]]                   // use default value count
    val me = me_default                                        // me-estimates
    val th = 0.0                                               // threshold

    println ("---------------------------------------------------------------")
    println ("D A T A   M A T R I X")
    println ("xy = " + xy)

    val nb     = BayesClassifier (xy, fn, k, cn, vc, me);     test (nb,     "Naive Bayes")
    val oneban = BayesClassifier (xy, fn, k, cn, vc, me, th); test (oneban, "1-BAN")
    val tan    = BayesClassifier (xy, fn, k, cn, me, vc);     test (tan,    "TAN Bayes")
    val twoban = BayesClassifier (xy, fn, k, cn, vc, th, me); test (twoban, "2-BAN-OS")

    nb.featureSelection ();     test (nb,     "Selective Naive Bayes")
    oneban.featureSelection (); test (oneban, "Selective 1-BAN")
    tan.featureSelection ();    test (tan,    "Selective TAN Bayes")
    twoban.featureSelection (); test (twoban, "Selective 2-BAN-OS")

} // BayesClassifierTest object