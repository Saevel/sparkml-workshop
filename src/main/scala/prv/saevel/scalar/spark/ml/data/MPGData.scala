package prv.saevel.scalar.spark.ml.data

import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.sql.{DataFrame, SparkSession}

import scala.io.Source

object MPGData {

  def apply(file: String)(implicit session: SparkSession): DataFrame = {
    import session.implicits._
    Source
      .fromFile(file)
      .getLines
      .map{line =>
        val split = line.split(",")
        (Vectors.dense(split(1).toDouble, split(2).toDouble, split(3).toDouble, split(4).toDouble, split(5).toDouble,
          split(6).toDouble, split(7).toDouble), split(0).toDouble, split(8))
      }
      .toSeq
      .toDF("features", "mpg", "name")
  }

  /**
    1. mpg:           continuous
    2. cylinders:     multi-valued discrete
    3. displacement:  continuous
    4. horsepower:    continuous
    5. weight:        continuous
    6. acceleration:  continuous
    7. model year:    multi-valued discrete
    8. origin:        multi-valued discrete
    9. car name:      string (unique for each instance)
    */
}
