package prv.saevel.scalar.spark.ml.utils

import org.apache.spark.ml.feature.{Binarizer, Bucketizer, StringIndexer, VectorAssembler}
import org.apache.spark.sql.{DataFrame, Dataset}
import prv.saevel.scalar.spark.ml.Patient

object PatientPreprocessor {
  /**
    * Preprocesses a <code>Dataset[Patient]</code>, by adding a "features" (Vector) column, constructed as follows:
    *
    * features(0) = height
    * features(1) = weigth
    * features(2) = sex field, indexed into Doubles
    * features(3) = age field, bucketized into the following buckets: [0, 10], [10, 20], [20, 40], [40, 70], [70, 100]
    * features(4) = 1.0 if "otherIndicators" > 0.75, 0.0 otherwise.
    */
  def preprocess(patients: Dataset[Patient]): DataFrame = ???
}