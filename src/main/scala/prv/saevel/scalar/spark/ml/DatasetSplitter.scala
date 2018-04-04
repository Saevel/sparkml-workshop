package prv.saevel.scalar.spark.ml

import org.apache.spark.sql.Dataset

object DatasetSplitter {

  def apply[T](dataset: Dataset[T], trainingFraction: Double): (Dataset[T], Dataset[T]) = {
    val splits = dataset.randomSplit(Array(trainingFraction, (1.0 - trainingFraction)))
    (splits(0), splits(1))
  }
}
