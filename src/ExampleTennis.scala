import scalation.linalgebra.{MatriD, MatrixD, VectoD, VectorD,VectorI,MatrixI}

object ExampleTennis
{
// dataset ----------------------------------------------------------------
// x0: Outlook: Rain (0), Overcast (1), Sunny (2)
// x1: Temperature: Cold (0), Mild (1), Hot (2)
// x2: Humidity: Normal (0), High (1)
// x3: Wind: Weak (0), Strong (1)
// y: the response/classification decision
// variables/features: x0 x1 x2 x3 y // combined data matrix
val xy = new MatrixI ((14, 5), 2, 2, 1, 0, 0, // day 1
2, 2, 1, 1, 0, // day 2
1, 2, 1, 0, 1, // day 3
0, 1, 1, 0, 1, // day 4
0, 0, 0, 0, 1, // day 5
0, 0, 0, 1, 0, // day 6
1, 0, 0, 1, 1, // day 7
2, 1, 1, 0, 0, // day 8
2, 0, 0, 0, 1, // day 9
0, 1, 0, 0, 1, // day 10
2, 1, 0, 1, 1, // day 11
1, 1, 1, 1, 1, // day 12
1, 2, 0, 0, 1, // day 13
0, 1, 1, 1, 0) // day 14
val fn = Array ("Outlook", "Temp", "Humidity", "Wind") // feature names
val cn = Array ("No", "Yes") // class names for y
val k = cn.size // number of classes
} // ExampleTennis object