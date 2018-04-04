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
  def preprocess(patients: Dataset[Patient]): DataFrame = {
    val sexIndexer = new StringIndexer().setInputCol("sex").setOutputCol("sex_indexed")
    val professionIndexer = new StringIndexer().setInputCol("occupation").setOutputCol("occupation_indexed")
    val bucketizer = new Bucketizer().setSplits(Array(0.0, 10.0, 20.0, 40.0, 70.0, 100.0)).setInputCol("age").setOutputCol("age_bucketized")
    val indicatorBinarizer = new Binarizer().setThreshold(0.75).setInputCol("otherIndicators").setOutputCol("indicators_binary")
    val assembler = new VectorAssembler()
      .setInputCols(Array("height", "weigth", "sex_indexed", "occupation_indexed", "age_bucketized", "indicators_binary"))
      .setOutputCol("features")

    val df1 = sexIndexer.fit(patients).transform(patients)
    val df2 = professionIndexer.fit(df1).transform(df1)
    val df3 = bucketizer.transform(df2)
    val df4 = indicatorBinarizer.transform(df3)
    assembler.transform(df4)
  }
}