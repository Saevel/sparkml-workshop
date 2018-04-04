package prv.saevel.scalar.spark.ml.utils

import org.apache.spark.ml.linalg._

case class PatientsPreprocessed(features: Vector, isIll: Boolean)
