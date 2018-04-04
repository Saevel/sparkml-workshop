package prv.saevel.scalar.spark.ml.crossvalidation

import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.RandomForestClassifier
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.{StringIndexerModel, VectorAssembler}
import org.apache.spark.ml.tuning.{CrossValidator, ParamGridBuilder}

object CrossValidatorPipeline {
  /**
    * Returns a <code>CrossValidator</code> with a <code>Pipeline</code> inside, which takes <code>IncomeData</code> elements
    * and build the "features" Vector as follows:
    *   features(0) = indexed "sex" field
    *   features(1) = indexed "educationLevel" field
    *   features(2) = indexed "fieldOfExpertise" field
    *   features(3) = age
    *   features(4) = yearsOfExperience
    *
    *   and then processes passes it through a <code>RandomForestClassifier</code>, with label column "incomeBracket"
    *   and prediction column "predicted_income_bracket".
    *
    *   The <code>CrossValidator</code> trains / validates the "numTrees" values for the <code>RandomForestClassifier</code>
    *   from the values in <code>possibleNumTress</code>.
    */
  def apply(possibleNumTrees: Array[Int]): CrossValidator = ???
}