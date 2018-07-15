import scalation.linalgebra.{MatriD, MatrixD, VectoD, VectorD,VectorI,MatrixI}
import scalation.math.double_exp
import scalation.plot.Plot
import scalation.util.Error
import scalation.analytics.classifier.BayesClassifier
import BayesClassifier.{me_default, test}
import scalation.analytics.classifier.ClassifierInt

object BayesClassifierTest2 extends App
{
    val BASE_DIR = "D:\\Users\\voghoei\\workspace\\scalation_1.4\\data\\analytics\\classifier\\"
    val fname = BASE_DIR + "bayes_data.csv"                      // file's relative path name
    val (m, n) = (683, 10)                                       // number of (rows/lines, columns) in file

    val xy = ClassifierInt (fname, m, n)                         // load 'xy' data matrix from file

    xy.setCol (n - 1, xy.col(n - 1).map((z: Int) => z / 2 - 1))  // transform the last column

    val fn = Array ("Clump Thickness", "Uniformity of Cell Size", "Uniformity of Cell Shape", "Marginal Adhesion",
                    "Single Epithelial Cell Size", "Bare Nuclei", "Bland Chromatin", "Normal Nucleoli", "Mitoses")
    val k  = 2
    val cn = Array ("benign", "malignant")
    val vc = Array (11, 11, 11, 11, 11, 11, 11, 11, 11)          // value count
    val me = me_default                                          // me-estimates
    val th = 0.0                                                 // threshold

    val nb     = BayesClassifier (xy, fn, k, cn, vc, me);     test (nb,     "Naive Bayes")
    val oneban = BayesClassifier (xy, fn, k, cn, vc, me, th); test (oneban, "1-BAN")
    val tan    = BayesClassifier (xy, fn, k, cn, me, vc);     test (tan,    "TAN Bayes")
    val twoban = BayesClassifier (xy, fn, k, cn, vc, th, me); test (twoban, "2-BAN-OS")

    nb.featureSelection ();     test (nb,     "Selective Naive Bayes")
    oneban.featureSelection (); test (oneban, "Selective 1-BAN")
    tan.featureSelection ();    test (tan,    "Selective TAN Bayes")
    twoban.featureSelection (); test (twoban, "Selective 2-BAN-OS")

} // BayesClassifierTest2 object