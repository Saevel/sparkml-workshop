// Importing the Spark ML Vector type
import org.apache.spark.ml.linalg._

// Dense Vector: traditional way of storing elements
val dense = Vectors.dense((0 to 27).map(i =>
  if(i % 9 == 0) i.toDouble else 0.0
).toArray)

// Sparse Vector: Storing length, indices of non-zero
// elements and their values
val sparse = Vectors.sparse(28, Seq((9, 9.0), (18, 18.0), (27, 27.0)))

// Vector operations
Vectors.sqdist(dense, sparse)

// Conversions dense <=> sparse
dense.toSparse

sparse.toDense


