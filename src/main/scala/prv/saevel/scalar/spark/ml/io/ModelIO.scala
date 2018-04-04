package prv.saevel.scalar.spark.ml.io

import org.apache.spark.ml.regression.LinearRegressionModel
import org.apache.spark.ml.util.MLWritable

object ModelIO {
  /**
    * Writes the model to a given file.
    * @param model model to save.
    * @param location location to which the model is saved.
    */
  def saveModel(model: MLWritable, location: String): Unit = ???

  /**
    * Reads a <code>LinearRegressionModel</code> from the given location.
    */
  def readLinearRegressionModel(location: String): LinearRegressionModel = ???

}
