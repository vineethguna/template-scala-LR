import io.prediction.controller.Params

case class AlgorithmParams(
  maxIter: Int,
  regParam: Double,
  fitIntercept: Boolean,
  categoricalPrediction: Boolean
) extends Params