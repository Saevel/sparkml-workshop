package prv.saevel.scalar.spark.ml.data

import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.sql.{DataFrame, SparkSession}

import scala.io.Source

object IrisDataset {

  def apply(file: String)(implicit session: SparkSession): (DataFrame, DataFrame) = {
    import session.implicits._

    val irisData = Source.fromFile(file)
      .getLines
      .map{ line =>
        val split = line.split(",")
        (Vectors.dense(Array(split(0).toDouble, split(1).toDouble, split(2).toDouble, split(3).toDouble)), split(4), split(4) match {
          case "Iris-setosa" => 0
          case "Iris-versicolor" => 1
          case "Iris-virginica" => 2
        })
      }.toSeq.toDF("features", "type", "type_numeric")

    val split = irisData.randomSplit(Array(0.1, 0.9))

    (split(1), split(0))
  }
}