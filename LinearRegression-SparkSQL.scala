// Databricks notebook source
// MAGIC %md
// MAGIC NORMAL EQUATION FOR LINEAR REGRESSION PROBLEM - SOLVING FOR MINUMUM COST FUNCTION PARAMETER THETA

// COMMAND ----------

// MAGIC %md
// MAGIC Steps:
// MAGIC 1. Create an example RDD for matrix X and vector y
// MAGIC 2. Compute \\[ \scriptsize \mathbf{(X^TX)}\\]
// MAGIC 3. Convert the result matrix to a Breeze Dense Matrix and compute pseudo-inverse
// MAGIC 4. Compute \\[ \scriptsize \mathbf{X^Ty}\\] and convert it to Breeze Vector
// MAGIC 5. Multiply \\[ \scriptsize \mathbf{(X^TX)}^{-1}\\] with \\[ \scriptsize \mathbf{X^Ty}\\]

// COMMAND ----------

// MAGIC %md
// MAGIC \\[ \scriptsize \mathbf{\theta=(X^TX)}^{-1}\mathbf{X^Ty}\\]

// COMMAND ----------

// MAGIC %md
// MAGIC IMPLEMENTATION USING SPARK SQL (DATAFRAME)

// COMMAND ----------

//Importing necessary libraries

import org.apache.spark.sql.{SparkSession, Row}
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types._
import org.apache.spark.sql.{DataFrame, SparkSession}
import breeze.linalg._
import breeze.numerics._

// Create a SparkSession
val spark = SparkSession.builder()
  .appName("RDDtoDataFrame")
  .getOrCreate()

// Define the dimensions of the matrix
val numRows = 3
val numCols = 3

// Create an example matrix X as an RDD
val rddMatrixX = spark.sparkContext.parallelize(
  Seq(
    ((0, 0), 1000.0), ((1, 0), 2000.0), ((2, 0), 3000.0),
     ((0,1),500.0), ((1,1),1000.0),((2,1),1500.0)
  )
)

// Create an example vector y as an RDD
val rddVectorY = spark.sparkContext.parallelize(Seq(100.0, 200.0, 300.0) )

// COMMAND ----------


// Create SparkSession
val spark = SparkSession.builder()
  .appName("RDDs to DataFrame")
  .getOrCreate()

// Define the schema for matrix X representing row, column and value
val matrixXSchema = StructType(
  Array(
    StructField("row", IntegerType, nullable = false),
    StructField("col", IntegerType, nullable = false),
    StructField("valueX", DoubleType, nullable = false)
  )
)

// Convert RDD rddMatrixX to DataFrame
val dfMatrixX = spark.createDataFrame(
  rddMatrixX.map { case ((row, col), value) => Row(row, col, value) },
  matrixXSchema
)

// Define the schema for vector Y
val vectorYSchema = StructType(
  Array(
    StructField("index", IntegerType, nullable = false),
    StructField("valueY", DoubleType, nullable = false)
  )
)

// Convert RDD rddVectorY to DataFrame
val dfVectorY = spark.createDataFrame(
  rddVectorY.zipWithIndex.map { case (value, index) => Row(index.toInt, value) },
  vectorYSchema
)

// Show the DataFrames
dfMatrixX.show()
dfVectorY.show()


// COMMAND ----------

//Find X transpose and persist in memory - Doing this will help reuse the value in multiplying X transpose and y

// Step 1: Convert DataFrame dfMatrixX into RDD of tuples
val rddMatrixXTuples = dfMatrixX.rdd.map { row =>
  val rowIdx = row.getInt(0)
  val colIdx = row.getInt(1)
  val value = row.getDouble(2)
  ((colIdx, rowIdx), value) // Swap row and column indices for transpose
}

// Step 2: Transpose the matrix by swapping the row and column indices
val rddTransposedMatrixX = rddMatrixXTuples.reduceByKey(_ + _)

// Step 3: Convert transposed RDD back into DataFrame
val dfTransposedMatrixX = spark.createDataFrame(
  rddTransposedMatrixX.map { case ((row, col), value) => Row(row, col, value) },
  matrixXSchema
)

// Step 4: Persist transposed matrix in memory to reuse with y
dfTransposedMatrixX.persist()

// Display the transposed DataFrame
dfTransposedMatrixX.show()


// COMMAND ----------

//Multiply X transpose and X


// Define a function for matrix multiplication using Spark SQL
def multiplyMatricesSQL(A: DataFrame, B: DataFrame, spark: SparkSession): DataFrame = {
    // Register the DataFrames as temporary views
    A.createOrReplaceTempView("A")
    B.createOrReplaceTempView("B")

    // Define the SQL query for matrix multiplication
    val query =
      """
        |SELECT A.row AS A_row, B.col AS B_col, SUM(A.valueX * B.valueX) AS value
        |FROM A
        |JOIN B ON A.col = B.row
        |GROUP BY A.row, B.col
      """.stripMargin

    // Execute the SQL query
    val result = spark.sql(query)

    // Return the resulting DataFrame
    result
}



// Multiply the matrices using Spark SQL
val result: DataFrame = multiplyMatricesSQL(dfTransposedMatrixX, dfMatrixX, spark)

// Display the result
result.show()

// COMMAND ----------

//Convert result into a Dense matrix using Breeze and find pseudoinverse of the result

val data = result.collect().map(row => (row.getInt(0), row.getInt(1), row.getDouble(2)))

// Determine the dimensions of the resulting matrix
val numRows = data.map(_._1).max + 1
val numCols = data.map(_._2).max + 1

// Construct a DenseMatrix from the collected data
val matrixData = Array.ofDim[Double](numRows, numCols)
for ((rowIdx, colIdx, value) <- data) {
  matrixData(rowIdx)(colIdx) = value
}

// Construct a DenseMatrix from the collected data
val denseMatrix = new DenseMatrix(numRows, numCols, matrixData.flatten)

// Compute the pseudoinverse using Breeze
val pseudoinverseMatrix = pinv(denseMatrix)



// COMMAND ----------

//Multiply X transpose and y
val transposedMatrix = dfTransposedMatrixX.collect() //Using the cached X transpose matrix from previous steps

// Register the DataFrames as temporary views
dfTransposedMatrixX.createOrReplaceTempView("transposedMatrix")
dfVectorY.createOrReplaceTempView("vectorY")

// Define the SQL query for matrix multiplication with vector
val query =
  """
    |SELECT t.row AS row, SUM(t.valueX * v.valueY) AS result
    |FROM transposedMatrix t
    |JOIN vectorY v ON t.col = v.index
    |GROUP BY t.row
  """.stripMargin

// Execute the SQL query
val result: DataFrame = spark.sql(query)

// Display the result
result.show()



// COMMAND ----------

// Converting the result to a dense vector

// Collect the data from the DataFrame as an array of tuples (row, result)
val collectedData = result.collect().map(row => (row.getInt(0), row.getDouble(1)))

// Create a DenseVector from the collected data
val breezeVector = DenseVector(collectedData.sortBy(_._1).map(_._2))

// Display the breeze vector
println("Breeze Vector:")
println(breezeVector)


// COMMAND ----------

//Multiplying the pseudoinverse and dense vector to get final theta value
val resultVector: DenseVector[Double] = pseudoinverseMatrix * breezeVector

// Display the resultant theta values
println("Result Vector:")
println(resultVector)

// COMMAND ----------

// MAGIC %md
// MAGIC USING THE ABOVE ALGORITHM ON A REAL-TIME DATASET - Boston Housing Dataset: https://www.kaggle.com/datasets/vikrishnan/boston-house-prices?resource=download
// MAGIC - As a result, we will be able to predict house price of large size dataset

// COMMAND ----------


val delimiter = "  " //Defining space as delimiter as per the csv to differentiate columns

// Read the CSV file into a DataFrame
val df = spark.read
  .format("csv")
  .option("header", "false") // Specify that the CSV file does not have a header row
  .option("sep", delimiter) // Specify the delimiter used in the CSV file
  .load("dbfs:/FileStore/shared_uploads/sanjana66982022@gmail.com/housing.csv")
// Show the DataFrame with default numbered columns
df.show()


// COMMAND ----------

//import necessary libraries
import org.apache.spark.sql.{SparkSession, Row}
import org.apache.spark.sql.types.{StructType, StructField, IntegerType, DoubleType}
import org.apache.spark.sql.functions._

// Separate the features and target
    
val columns = df.columns
val targetColumn = columns.last

// Create DataFrame for attributes (features) by dropping the last column
    val dfAttributes = df.drop(targetColumn)

// Ensure data types are inferred when reading CSV
val dfAttributesWithInferredSchema = spark.read
.option("inferSchema", "true")
.csv("dbfs:/FileStore/shared_uploads/sanjana66982022@gmail.com/housing.csv")

// Convert the features tributes DataFrame to an RDD
val rddAttributes = dfAttributes.rdd

dfAttributes.show()
    

// COMMAND ----------

val dfAttributes = df.drop(targetColumn)

// Convert all columns in dfAttributes to Double type
val dfAttributesConverted = dfAttributes.select(dfAttributes.columns.map(c => col(c).cast("double")): _*)

// Convert the DataFrame to an RDD
val rddAttributes = dfAttributesConverted.rdd

// Define the matrixXSchema
val matrixXSchema = StructType(
    Array(
        StructField("row", IntegerType, nullable = false),
        StructField("col", IntegerType, nullable = false),
        StructField("valueX", DoubleType, nullable = false)
    )
)

// Transform the RDD and create dfMatrixX
val rddMatrixX = rddAttributes.flatMap { row =>
    row.toSeq.zipWithIndex.map { case (value, colIndex) =>
        ((0, colIndex), value.asInstanceOf[Double])
    }
}

val dfMatrixXBoston = spark.createDataFrame(
    rddMatrixX.map { case ((row, col), value) => Row(row, col, value) },
    matrixXSchema
)

// Show the DataFrame
dfMatrixXBoston.show()


// COMMAND ----------

// Get the last column name
val lastColumnIndex = df.columns.length - 1
val lastColumnName = df.columns(lastColumnIndex)

// Convert the last column to Double type
val dfLastColumnAsDouble = df.withColumn(lastColumnName, col(lastColumnName).cast(DoubleType))

// Handle null values in the last column by filling them with a default 0.0
val dfLastColumnAsDoubleFilled = dfLastColumnAsDouble.na.fill(0.0, Array(lastColumnName))


// Extract the last column as an RDD
val rddVectorY = dfLastColumnAsDoubleFilled.select(lastColumnName).rdd.map(row => row.getDouble(0))

// Zip the RDD with an index
val rddVectorYWithIndex = rddVectorY.zipWithIndex.map { case (value, index) => Row(index.toInt, value) }

// Define the schema for vectorY
val vectorYSchema = StructType(
  Array(
    StructField("index", IntegerType, nullable = false),
    StructField("valueY", DoubleType, nullable = false)
  )
)

// Convert the RDD to a DataFrame using the schema
val dfVectorY = spark.createDataFrame(rddVectorYWithIndex, vectorYSchema)

// Show the DataFrame
dfVectorY.show()

// COMMAND ----------

import org.apache.spark.sql.functions._
import org.apache.spark.sql.types._

//Find X transpose and persist in memory - Doing this will help reuse the value in multiplying X transpose and y

// Step 1: Convert DataFrame dfMatrixX into RDD of tuples
val rddMatrixXTuples = dfMatrixXBoston.rdd.map { row =>
  val rowIdx = row.getInt(0)
  val colIdx = row.getInt(1)
  val value = row.getDouble(2)
  ((colIdx, rowIdx), value) // Swap row and column indices for transpose
}

// Step 2: Transpose the matrix by swapping the row and column indices
 val rddTransposedMatrixX = rddMatrixXTuples

// Step 3: Convert transposed RDD back into DataFrame
val dfTransposedMatrixXBoston = spark.createDataFrame(
  rddTransposedMatrixX.map { case ((row, col), value) => Row(row, col, value) },
  matrixXSchema
)

// Step 4: Persist transposed matrix in memory to reuse with y
//dfTransposedMatrixXBoston.persist()

// Display the transposed DataFrame
dfTransposedMatrixXBoston.show()



// COMMAND ----------

// Multiply the matrices using Spark SQL
val resultBoston: DataFrame = multiplyMatricesSQL(dfTransposedMatrixXBoston, dfMatrixXBoston, spark)

// Display the result
resultBoston.show()

// COMMAND ----------

//Convert result into a Dense matrix using Breeze and find pseudoinverse of the result

val data = resultBoston.collect().map(row => (row.getInt(0), row.getInt(1), row.getDouble(2)))

// Determine the dimensions of the resulting matrix
val numRows = data.map(_._1).max + 1
val numCols = data.map(_._2).max + 1

// Construct a DenseMatrix from the collected data
val matrixData = Array.ofDim[Double](numRows, numCols)
for ((rowIdx, colIdx, value) <- data) {
  matrixData(rowIdx)(colIdx) = value
}

// Construct a DenseMatrix from the collected data
val denseMatrixBoston = new DenseMatrix(numRows, numCols, matrixData.flatten)

// Compute the pseudoinverse using Breeze
val pseudoinverseMatrixBoston = pinv(denseMatrixBoston)



// COMMAND ----------

//Multiply X transpose and y
val transposedMatrixBoston = dfTransposedMatrixXBoston.collect() //Using the cached X transpose matrix from previous steps

// Register the DataFrames as temporary views
dfTransposedMatrixXBoston.createOrReplaceTempView("transposedMatrix")
dfVectorY.createOrReplaceTempView("vectorY")

// Execute the SQL query
val resultBostonXty: DataFrame = spark.sql(query)

// Display the result
resultBostonXty.show()



// COMMAND ----------

// Converting the result to a dense vector

// Collect the data from the DataFrame as an array of tuples (row, result)
val collectedData = resultBostonXty.collect().map(row => (row.getInt(0), row.getDouble(1)))

// Create a DenseVector from the collected data
val breezeVectorBoston = DenseVector(collectedData.sortBy(_._1).map(_._2))

// Display the breeze vector
println("Breeze Vector:")
println(breezeVectorBoston)


// COMMAND ----------

//Multiplying the pseudoinverse and dense vector to get final theta value
val resultVector: DenseVector[Double] = pseudoinverseMatrixBoston * breezeVectorBoston

// Display the resultant theta values
println("Result Vector:")
println(resultVector)
