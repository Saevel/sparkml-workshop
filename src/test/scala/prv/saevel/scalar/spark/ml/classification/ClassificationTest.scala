package prv.saevel.scalar.spark.ml.classification

import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.sql.SparkSession
import org.junit.runner.RunWith
import org.scalatest.junit.JUnitRunner
import org.scalatest.{Matchers, WordSpec}
import prv.saevel.scalar.spark.ml.data.IrisDataset


@RunWith(classOf[JUnitRunner])
class ClassificationTest extends WordSpec with Matchers {

  implicit val session = SparkSession.builder.master("local[*]").appName("ClassificationApplication").getOrCreate

  "Classification" should {

    "classify Iris data with reasonable efficiency" in {

      // Data with columns: "features", "type_numeric", "type"
      val (learningData, checkData) = IrisDataset("src/main/resources/iris.txt")

      val results = Classification(learningData, checkData)

      // Showing the results with predictions.
      println("RESULTS: ")
      results.show

      // Evaluator to judge the accuracy of the prediction
      val evaluator = new MulticlassClassificationEvaluator().setLabelCol("type_numeric").setPredictionCol("type_predicted")

      val efficiency = evaluator.evaluate(results)

      println(s"CLASSIFIER EFFICIENCY: ${efficiency}")

      efficiency should be > 0.5
    }
  }
}