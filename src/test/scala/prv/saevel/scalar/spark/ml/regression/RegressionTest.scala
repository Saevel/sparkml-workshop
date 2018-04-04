package prv.saevel.scalar.spark.ml.regression

import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.regression.LinearRegression
import org.apache.spark.sql.SparkSession
import org.junit.runner.RunWith
import org.scalatest.junit.JUnitRunner
import org.scalatest.{Matchers, WordSpec}
import prv.saevel.scalar.spark.ml.DatasetSplitter
import prv.saevel.scalar.spark.ml.data.MPGData

@RunWith(classOf[JUnitRunner])
class RegressionTest extends WordSpec with Matchers {

  implicit val session = SparkSession.builder.master("local[*]").appName("RegressionApplication").getOrCreate

  val mpgData = MPGData("src/main/resources/mpg.txt")

  val dataArray = mpgData.randomSplit(Array(0.1, 0.9))

  val (learningData, checkData) = DatasetSplitter(mpgData, 0.9)

  "Regression" should {

    "correctly predict mpg values for the most part" in {

      val results = Regression(learningData, checkData)

      println("RESULTS:")
      results.show

      val evaluator = new RegressionEvaluator().setLabelCol("mpg").setPredictionCol("mpg_predicted")

      val error = evaluator.evaluate(results)

      error should be < 4.0

      println(s"REGRESSION ERROR: $error")
    }
  }
}