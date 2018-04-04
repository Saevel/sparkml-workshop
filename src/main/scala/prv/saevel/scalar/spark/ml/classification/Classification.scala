package prv.saevel.scalar.spark.ml.classification

import org.apache.spark.ml.classification.DecisionTreeClassifier
import org.apache.spark.sql.DataFrame

object Classification {
  /**
    * Learns a <code>DecisionTreeClassifier</code> on <code>trainingData</code> - a <code>DataFrame</code> with features
    * in the "features" column and labels in the "type_numeric" column and then transforms <code>verificationData</code>
    * adding predicted classes as the "type_predicted" column.
    */
  def apply(trainingData: DataFrame, verificationData: DataFrame): DataFrame =
    new DecisionTreeClassifier()
      .setLabelCol("type_numeric")
      .setFeaturesCol("features")
      .setPredictionCol("type_predicted")
      .fit(trainingData)
      .transform(verificationData)
}
