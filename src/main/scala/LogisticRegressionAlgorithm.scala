import io.prediction.controller.P2LAlgorithm
import io.prediction.controller.Params

import org.apache.spark.mllib.classification.LogisticRegressionWithLBFGS
import org.apache.spark.mllib.classification.LogisticRegressionModel
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.SparkContext

import grizzled.slf4j.Logger

case class AlgorithmParams(
  maxIter: Int,
  regParam: Double,
  fitIntercept: Boolean,
  categoricalPrediction: Boolean
) extends Params

// extends P2LAlgorithm because the MLlib's NaiveBayesModel doesn't contain RDD.
class LogisticRegressionAlgorithm()
  extends P2LAlgorithm[PreparedData, LogisticRegressionModel, Query, PredictedResult] {

  @transient lazy val logger = Logger[this.type]

  def train(sc: SparkContext, data: PreparedData): LogisticRegressionModel = {
    // MLLib NaiveBayes cannot handle empty training data.
    require(data.labeledPoints.take(1).nonEmpty,
      s"RDD[labeledPoints] in PreparedData cannot be empty." +
      " Please check if DataSource generates TrainingData" +
      " and Preparator generates PreparedData correctly.")
    new LogisticRegressionWithLBFGS().run(data.labeledPoints)
    //NaiveBayes.train(data.labeledPoints, ap.lambda)
  }

  def predict(model: LogisticRegressionModel, query: Query): PredictedResult = {
    val label = model.predict(Vectors.dense(query.features))
    new PredictedResult(label)
  }

}
