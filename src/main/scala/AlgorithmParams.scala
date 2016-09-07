case class AlgorithmParams(
  maxIter: Int,
  regParam: Double,
  fitIntercept: Boolean,
  categoricalPrediction: Boolean
) extends Params