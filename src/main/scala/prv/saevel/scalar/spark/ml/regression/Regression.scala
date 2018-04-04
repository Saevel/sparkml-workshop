package prv.saevel.scalar.spark.ml.regression

import org.apache.spark.ml.regression.LinearRegression
import org.apache.spark.sql.DataFrame

object Regression {
  /**
    * Applies <code>LinearRegression</code> based on the "features" column and the "mpg" label to <code>trainingData</code>
    * and then transforms <code>verificationData</code> adding predictions to the "mpg_predicted" column.
    */
  def apply(trainingData: DataFrame, verificationData: DataFrame): DataFrame = ???
}
