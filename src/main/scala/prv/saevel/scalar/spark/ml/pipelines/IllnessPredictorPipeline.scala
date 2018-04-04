package prv.saevel.scalar.spark.ml.pipelines

import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.DecisionTreeClassifier
import org.apache.spark.ml.feature._

object IllnessPredictorPipeline {
  /**
    * Constructs an ML Pipeline that will transform the data into a Vector form names "features" defined as:
    * features(0) = sex field indexed
    * features(1) = weigth
    * features(2) = height
    * features(3) = otherIndicators + otherIndicators^2 + otherIndicators^3
    * features(4) = age + age^2 + age^3
    * and use a <code>DecisionTreeClassifier</code> to predict whether the patient is ill and puts them in the
    * "illness_predicted" column.
    */
  def apply(): Pipeline = {
    new Pipeline().setStages(Array(
      new StringIndexer().setInputCol("sex").setOutputCol("sex_indexed").setHandleInvalid("keep"),
      new VectorAssembler().setInputCols(Array("age", "otherIndicators")).setOutputCol("preliminary_vector"),
      new PolynomialExpansion().setDegree(3).setInputCol("preliminary_vector").setOutputCol("preliminary_poly"),
      new VectorAssembler().setInputCols(Array("sex_indexed", "weigth", "height", "preliminary_poly")).setOutputCol("features"),
      new DecisionTreeClassifier().setPredictionCol("illness_prediction").setFeaturesCol("features").setLabelCol("illness_actual")
    ))
  }
}