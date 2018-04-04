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
  def apply(possibleNumTrees: Array[Int]): CrossValidator = {
    val classifier = new RandomForestClassifier().setFeaturesCol("features").setPredictionCol("predicted_income_bracket").setLabelCol("incomeBracket")

    val evaluator = new MulticlassClassificationEvaluator().setPredictionCol("predicted_income_bracket").setLabelCol("incomeBracket").setMetricName("accuracy")

    val pipeline = new Pipeline().setStages(Array(
      new StringIndexerModel(Array("Female", "Male")).setInputCol("sex").setOutputCol("sex_indexed"),
      new StringIndexerModel(Array("Primary", "Secondary", "Higher", "PhD")).setInputCol("educationLevel").setOutputCol("education_indexed"),
      new StringIndexerModel(Array("Physical", "Retail", "STEM", "Medical")).setInputCol("fieldOfExpertise").setOutputCol("field_indexed"),
      new VectorAssembler().setInputCols(Array("sex_indexed", "education_indexed", "field_indexed", "age", "yearsOfExperience")).setOutputCol("features"),
      classifier
    ))

    new CrossValidator()
      .setEstimator(pipeline)
      .setEstimatorParamMaps(new ParamGridBuilder().addGrid(classifier.numTrees, possibleNumTrees).build)
      .setEvaluator(evaluator)
  }
}