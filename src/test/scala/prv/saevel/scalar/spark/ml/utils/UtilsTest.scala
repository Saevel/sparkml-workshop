package prv.saevel.scalar.spark.ml.utils

import org.apache.spark.sql.SparkSession
import org.junit.runner.RunWith
import org.scalacheck.Gen
import org.scalatest.junit.JUnitRunner
import org.scalatest.prop.PropertyChecks
import org.scalatest.{Matchers, WordSpec}
import prv.saevel.scalar.spark.ml.Patient

@RunWith(classOf[JUnitRunner])
class UtilsTest extends WordSpec with PropertyChecks with Matchers {

  implicit val session = SparkSession.builder.master("local[1]").appName("UtilsTest").getOrCreate

  val patients: Gen[List[Patient]] = Gen.choose(10, 50).flatMap(n => Gen.listOfN(n, for {
    sex <- Gen.oneOf(Seq("Male", "Female"))
    age <- Gen.choose(5.0, 90.0)
    height <- Gen.choose(100.0, 200.0)
    weigth <- Gen.choose(20.0, 120.0)
    profession <- Gen.oneOf("Clerk", "Miner", "Pilot", "Office Worker", "Programmer")
    otherIndicators <- Gen.choose(0.0, 1.0)
    isIll <- Gen.oneOf(Seq(true, false))
  } yield Patient(sex, age, height, weigth, profession, otherIndicators, isIll)))

  "PatientPreprocessor" should {

    "perform the required data preprocessing" in forAll(patients){ patients =>

      import session.implicits._

      val patientsDS = patients.toDS

      PatientPreprocessor.preprocess(patientsDS).as[PatientsPreprocessed].show
    }
  }
}
