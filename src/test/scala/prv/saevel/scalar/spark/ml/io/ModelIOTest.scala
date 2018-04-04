package prv.saevel.scalar.spark.ml.io

import java.util.UUID

import org.apache.spark.ml.linalg.{Matrices, Matrix, Vectors}
import org.apache.spark.ml.classification.LogisticRegressionModel
import org.apache.spark.ml.clustering.LDAModel
import org.apache.spark.ml.regression.LinearRegression
import org.apache.spark.sql.SparkSession
import org.junit.runner.RunWith
import org.scalacheck.Gen
import org.scalatest.{Matchers, WordSpec}
import org.scalatest.junit.JUnitRunner
import org.scalatest.prop.PropertyChecks

@RunWith(classOf[JUnitRunner])
class ModelIOTest extends WordSpec with Matchers with PropertyChecks {

  private val coefficients: Gen[Double] = Gen.choose(0.01, 7.0)

  implicit val session = SparkSession.builder.master("local[*]").appName("ModelIOTest").getOrCreate

  "ModelIO" should {

    "save and model read back the same" in forAll(coefficients){ a =>

      import session.implicits._

      val data = (0.0 to 100.0 by 0.1).map(x =>
        (a * x, Vectors.dense(Array(x)))
      ).toDF("y", "x")

      val model = new LinearRegression().setLabelCol("y").setFeaturesCol("x").fit(data)

      ModelIO.saveModel(model, "build/models/example")

      val retrieved = ModelIO.readLinearRegressionModel("build/models/example")

      retrieved.params should equal(model.params)
    }
  }
}