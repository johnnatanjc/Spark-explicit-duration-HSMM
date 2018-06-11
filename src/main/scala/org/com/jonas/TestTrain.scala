package org.com.jonas

import java.io.FileInputStream

import breeze.linalg.{DenseMatrix, DenseVector, normalize}
import org.apache.log4j.{Level, Logger}
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.sql.functions._
import org.apache.spark.sql.expressions.{UserDefinedFunction, Window}

object TestTrain {
  def main(args: Array[String]): Unit = {
    val log = org.apache.log4j.LogManager.getRootLogger
    val applicationProps = new java.util.Properties()
    val in = new FileInputStream(args(0))
    applicationProps.load(in)
    in.close()

    if (applicationProps.getProperty("generate_logs").equals("true")) {
      Logger.getLogger("org").setLevel(Level.ERROR)
      Logger.getLogger("akka").setLevel(Level.ERROR)
    }

    val sparkSession = SparkSession.builder.appName("Spark-HSMM").getOrCreate()

    /**
      * Class 1 seq with error
      * Class 0 seq without error
      */
    val k_folds = applicationProps.getProperty("k_folds").toInt
    log.info("Value of k_folds: " + k_folds)
    val value_M = applicationProps.getProperty("value_M").toInt
    log.info("Value of value_M: " + value_M)
    val value_k = applicationProps.getProperty("value_k").toInt
    log.info("Value of value_k: " + value_k)
    val value_D = applicationProps.getProperty("value_D").toInt
    log.info("Value of value_D: " + value_D)
    val number_partitions = applicationProps.getProperty("number_partitions").toInt
    log.info("Value of number_partitions: " + number_partitions)
    val value_epsilon = applicationProps.getProperty("value_epsilon").toDouble
    log.info("Value of value_epsilon: " + value_epsilon)
    val max_num_iterations = applicationProps.getProperty("max_num_iterations").toInt
    log.info("Value of max_num_iterations: " + max_num_iterations)

    var sampleClass1 = sparkSession.read.csv(applicationProps.getProperty("path_sample_Class1"))
      .sample(withReplacement = false, applicationProps.getProperty("size_sample").toDouble)
      .withColumnRenamed("_c0", "workitem").withColumnRenamed("_c1", "str_obs")
      .select(col("workitem"), col("str_obs"), row_number().over(Window.orderBy(col("workitem"))).alias("rowId"))
      .where("rowId == 1")

    val tmpModelClass1 = hsmm.BaumWelchAlgorithm.run2(sampleClass1, value_M, value_k, value_D,
      normalize(DenseVector.rand(value_M), 1.0),
      hsmm.Utils.mkstochastic(DenseMatrix.rand(value_M, value_M)),
      hsmm.Utils.mkstochastic(DenseMatrix.rand(value_M, value_k)),
      hsmm.Utils.mkstochastic(DenseMatrix.rand(value_M, value_D)),
      number_partitions, value_epsilon, max_num_iterations,
      0, applicationProps.getProperty("path_result_Class1_models_baumwelch"))

    sampleClass1.show()

    sparkSession.stop()
  }

}
