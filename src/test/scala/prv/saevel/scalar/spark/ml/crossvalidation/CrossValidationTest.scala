package prv.saevel.scalar.spark.ml.crossvalidation

import org.apache.spark.sql.SparkSession
import org.junit.runner.RunWith
import org.scalacheck.Gen
import org.scalatest.junit.JUnitRunner
import org.scalatest.prop.PropertyChecks
import org.scalatest.{Matchers, WordSpec}
import prv.saevel.scalar.spark.ml.DatasetSplitter

@RunWith(classOf[JUnitRunner])
class CrossValidationTest extends WordSpec with PropertyChecks with Matchers {

  implicit val params: PropertyCheckConfiguration = PropertyCheckConfiguration(minSuccessful = 10)

  implicit val session = SparkSession.builder.appName("CrossValidationTest").master("local[1]").getOrCreate

  private val incomeData: Gen[List[IncomeData]] = Gen.choose(100, 150).flatMap(n => Gen.listOfN(n, for {
    age <- Gen.choose(20, 80)
    sex <- Gen.oneOf("Male", "Female")
    educationLevel <- Gen.oneOf("Primary", "Secondary", "Higher", "PhD")
    fieldOfExpertise <- educationLevel match {
      case "Primary" => Gen.oneOf("Physical", "Retail")
      case "Secondary" => Gen.oneOf("Physical", "Retail", "Medical")
      case "Higher" => Gen.oneOf("Retail", "Medical", "STEM")
      case "PhD" => Gen.oneOf("Medical", "STEM")
    }
    yearsOfExperience <- Gen.choose(0, age - 20)
  } yield IncomeData(
    age.toDouble,
    sex, educationLevel,
    fieldOfExpertise,
    yearsOfExperience.toDouble,
    predictIncome(age.toDouble, sex, educationLevel, fieldOfExpertise, yearsOfExperience.toDouble),
    selectBracket(predictIncome(age.toDouble, sex, educationLevel, fieldOfExpertise, yearsOfExperience.toDouble))
  )))

  "CrossValidatorPipeline" should {

    "find optimal numTrees on the test dataset and use it" in forAll(incomeData){ incomeData =>

      import session.implicits._

      val incomeDS = incomeData.toDS

      val (learningData, verificationData) = DatasetSplitter.apply(incomeDS, 0.9)

      val results = CrossValidatorPipeline((5 to 15).toArray).fit(learningData).transform(verificationData)

      results.show

      if(results.count > 0) {
        val verification = results.select($"incomeBracket" === $"predicted_income_bracket").as[Boolean].collect()

        val score = verification.count(x => x).toDouble / verification.size

        score should be > 0.5
      }
    }
  }

  private def predictIncome(age: Double, sex: String, educationLevel: String, fieldOfExpertise: String, yearsOfExperience: Double): Double = {
    20.0 * (yearsOfExperience + yearsOfExperience * yearsOfExperience) + (sex match {
      case "Male" => 500.0
      case "Female" => 0.0
    }) + (educationLevel match {
      case "Primary" => 150.0
      case "Secondary" => 300.0
      case "Higher" => 700.0
      case "PhD" => 1000.0
    }) + ( fieldOfExpertise match {
      case "Physical" => 250.0
      case "Retail" => 500.0
      case "Medical" => 2000.0
      case "STEM" => 3000.0
    }) + 75.0 * (0.8 * age - 0.01 * age * age + 15)
  }

  private def selectBracket(income: Double): Double = {
    if(income < 4000.0) {
      1.0
    } else if(income < 6000.0) {
      2.0
    } else if(income < 8000.0) {
      3.0
    } else {
      4.0
    }
  }
}
