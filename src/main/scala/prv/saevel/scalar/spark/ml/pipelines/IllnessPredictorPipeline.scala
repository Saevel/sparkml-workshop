package prv.saevel.scalar.spark.ml.pipelines

import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.DecisionTreeClassifier
import org.apache.spark.ml.feature._

object IllnessPredictorPipeline {
  /**
    * Constructs an ML Pipeline that will transform the data into a Vector form names "features" and use a
    * <code>DecisionTreeClassifier</code> to predict the values of "illness_actual" label and puts them in the
    * "illness_actual" column.
    */
  def apply(): Pipeline = ???
}