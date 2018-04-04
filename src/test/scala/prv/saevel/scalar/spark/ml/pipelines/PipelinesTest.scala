package prv.saevel.scalar.spark.ml.pipelines

import org.apache.spark.ml.evaluation.{BinaryClassificationEvaluator, Evaluator}
import org.apache.spark.sql.SparkSession
import org.junit.runner.RunWith
import org.scalacheck.{Gen, Shrink}
import org.scalatest.junit.JUnitRunner
import org.scalatest.{Matchers, WordSpec}
import org.scalatest.prop.PropertyChecks
import prv.saevel.scalar.spark.ml.{DatasetSplitter, Patient}

@RunWith(classOf[JUnitRunner])
class PipelinesTest extends WordSpec with PropertyChecks with Matchers {

  private implicit val config: PropertyCheckConfiguration = PropertyCheckConfiguration(minSuccessful = 50)

  private implicit val session = SparkSession.builder.master("local[1]").appName("PipelinesTest").getOrCreate

  implicit val noShrink: Shrink[(Double, Double, List[Patient])] = Shrink.shrinkAny

  protected case class RawData(sex: String, age: Double, height: Double, weigth: Double, profession: String, otherIndicators: Double, mainIndicator: Double)

  private val rawDataGenerator: Gen[List[RawData]] = Gen.choose(100, 500).flatMap(n => Gen.listOfN(n, for {
    sex <- Gen.frequency((6, Gen.const("Male")), (4, Gen.const("Female")))
    age <- Gen.choose(10, 100).map(_.toDouble)
    height <- Gen.choose(100.0, 200.0)
    weigth <- Gen.choose(20.0, 120.0)
    profession <- Gen.oneOf("Clerk", "Miner", "Pilot", "Office Worker", "Programmer")
    otherIndicators <- Gen.choose(0.0, 1.0)
  } yield RawData(sex, age, height, weigth, profession, otherIndicators, calculateProbability(sex, age, height, weigth, profession, otherIndicators))))

  private val patients: Gen[(Double, Double, List[Patient])] = rawDataGenerator.map{ rawData =>
    val minIndicator = rawData.map(_.mainIndicator).min
    val maxIndicator = rawData.map(_.mainIndicator).max
    val indicatorInterval = maxIndicator - minIndicator
    (minIndicator, maxIndicator, rawData.map(data =>
      Patient(data.sex, data.age, data.height, data.weigth, data.profession, data.otherIndicators,
        if((data.mainIndicator - minIndicator) / indicatorInterval >= 0.75) true else false)
    ))
  }

  "Given a collection of patients IllnessPredictorPipeline" should  {

    "mostly correctly predict the probability of illness" in forAll(patients){ case (minIndicator, maxIndicator, patients) =>

        import session.implicits._
        import org.apache.spark.sql.functions._

        val indicatorInterval = maxIndicator - minIndicator

        val patientsDS = patients.toDS.withColumn("illness_actual", when($"isIll" === true, 1.0).otherwise(0.0))

        val (trainingData, testData) = DatasetSplitter(patientsDS, 0.9)

        val results = IllnessPredictorPipeline().fit(trainingData).transform(testData).cache

        println("RESULTS")
        results.show

        if(testData.count > 0) {
          val evaluations = results.select($"illness_actual" === $"illness_prediction").as[Boolean].collect
          (evaluations.count(x => x).toDouble / evaluations.size) should be > 0.5
        }
    }
  }

  private def calculateProbability(sex: String, age: Double, height: Double, weigth: Double, profession: String, other: Double): Double = {
    (0.1 * weigth + 0.005 * height + 0.75 * (age * age * age) + 0.85 * (other * other * other) + (sex match {
      case "Male" => 0.3
      case "Female" => 0.0
    }))
  }
}