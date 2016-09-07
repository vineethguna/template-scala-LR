import io.prediction.controller.P2LAlgorithm
import io.prediction.controller.Params

import org.apache.spark.mllib.classification.SVMWithSGD
import org.apache.spark.mllib.classification.SVMModel
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.SparkContext

import grizzled.slf4j.Logger

// extends P2LAlgorithm because the MLlib's NaiveBayesModel doesn't contain RDD.
class LinearSVM()
  extends P2LAlgorithm[PreparedData, SVMModel, Query, PredictedResult] {

  @transient lazy val logger = Logger[this.type]

  def train(sc: SparkContext, data: PreparedData): SVMModel = {
    // MLLib NaiveBayes cannot handle empty training data.
    require(data.labeledPoints.take(1).nonEmpty,
      s"RDD[labeledPoints] in PreparedData cannot be empty." +
      " Please check if DataSource generates TrainingData" +
      " and Preparator generates PreparedData correctly.")
    SVMWithSGD.train(data.labeledPoints, 100)
    //NaiveBayes.train(data.labeledPoints, ap.lambda)
  }

  def predict(model: SVMModel, query: Query): PredictedResult = {
    val label = model.predict(Vectors.dense(query.features))
    new PredictedResult(label)
  }

}
