package org.com.jonas

import java.io.FileInputStream

import breeze.linalg.{DenseMatrix, DenseVector, normalize}
import org.apache.log4j.{Level, Logger}
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.sql.functions._
import org.apache.spark.sql.expressions.{UserDefinedFunction, Window}

object CrossValidationWithLength {
  /**
    * @param args
    * args(0): Config Properties File
    * @return result
    *
    */
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
    sparkSession.conf.set("spark.speculation","false")
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
    val value_T = applicationProps.getProperty("value_T").toInt
    log.info("Value of value_T: " + value_T)
    val number_partitions = applicationProps.getProperty("number_partitions").toInt
    log.info("Value of number_partitions: " + number_partitions)
    val value_epsilon = applicationProps.getProperty("value_epsilon").toDouble
    log.info("Value of value_epsilon: " + value_epsilon)
    val max_num_iterations = applicationProps.getProperty("max_num_iterations").toInt
    log.info("Value of max_num_iterations: " + max_num_iterations)

    var sampleClass1 = sparkSession.emptyDataFrame
    var sampleClass0 = sparkSession.emptyDataFrame
    var nClass1 = 0
    var nClass0 = 0
    var inInter = 0

    sampleClass1 = sparkSession.read.csv(applicationProps.getProperty("path_sample_Class1_folds"))
      .withColumnRenamed("_c0", "workitem")
      .withColumnRenamed("_c1", "str_obs")
      .withColumnRenamed("_c2", "rowId")
      .withColumnRenamed("_c3", "kfold")
    nClass1 = sampleClass1.count().toInt
    log.info("Value of nClass1: " + nClass1)

    sampleClass0 = sparkSession.read.csv(applicationProps.getProperty("path_sample_Class0_folds"))
      .withColumnRenamed("_c0", "workitem")
      .withColumnRenamed("_c1", "str_obs")
      .withColumnRenamed("_c2", "rowId")
      .withColumnRenamed("_c3", "kfold")
    nClass0 = sampleClass0.count().toInt
    log.info("Value of nClass0: " + nClass0)

    inInter = scala.io.Source.fromFile(applicationProps.getProperty("path_result")).getLines.size - 1

    sampleClass1.persist()
    sampleClass0.persist()

    var validClass1 = sparkSession.emptyDataFrame
    var validClass0 = sparkSession.emptyDataFrame

    (inInter until k_folds).foreach(inter => {
      log.info("*****************************************************************************************")
      log.info("Fold number: " + inter)
      //log.info("Getting data to train Class 1")
      //val trainClass1 = sampleClass1.where("kfold <> " + inter).drop("kfold", "rowId")
      //log.info("Getting data to train Class 0")
      //val trainClass0 = sampleClass0.where("kfold <> " + inter).drop("kfold", "rowId")
      log.info("Getting data to validate Class 1")
      validClass1 = sampleClass1.where("kfold == " + inter).drop("kfold", "rowId")
      log.info("Getting data to validate Class 0")
      validClass0 = sampleClass0.where("kfold == " + inter).drop("kfold", "rowId")
      log.info("*****************************************************************************************")

      var modelClass1 = (Array.empty[Double], Array.empty[Double], Array.empty[Double], Array.empty[Double])
      var modelClass0 = (Array.empty[Double], Array.empty[Double], Array.empty[Double], Array.empty[Double])

      log.info("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
      log.info("Start Load Model Class 1")
      val stringModel1: List[String] = scala.io.Source.fromFile(applicationProps.getProperty("path_result_Class1_models")).getLines().toList
      val arraymodel1 = stringModel1(inter + 1).split(";")
      modelClass1 = (arraymodel1(4).split(",").map(_.toDouble),
        arraymodel1(5).split(",").map(_.toDouble),
        arraymodel1(6).split(",").map(_.toDouble),
        arraymodel1(7).split(",").map(_.toDouble))
      log.info("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")

      log.info("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
      log.info("Start Load Model Class 0")
      val stringModel0: List[String] = scala.io.Source.fromFile(applicationProps.getProperty("path_result_Class0_models")).getLines().toList
      val arraymodel0 = stringModel0(inter + 1).split(";")
      modelClass0 = (arraymodel0(4).split(",").map(_.toDouble),
        arraymodel0(5).split(",").map(_.toDouble),
        arraymodel0(6).split(",").map(_.toDouble),
        arraymodel0(7).split(",").map(_.toDouble))
      log.info("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")

      var resultClass1   = sparkSession.emptyDataFrame
      var resultClass0   = sparkSession.emptyDataFrame

      if (new java.io.File(applicationProps.getProperty("path_sample_Class1_kfold") + inter).exists) {
        log.info("Loading result validation Class1")
        resultClass1 = sparkSession.read.csv(applicationProps.getProperty("path_sample_Class1_kfold") + inter)
          .withColumnRenamed("_c0", "workitem")
          .withColumnRenamed("_c1", "probMod1")
          .withColumnRenamed("_c2", "probMod0")

      }else{
        resultClass1 =
          hsmm.BaumWelchAlgorithm.validateLength(validClass1, value_M, value_k, value_D, value_T,
            new DenseVector(modelClass1._1), new DenseMatrix(value_M, value_M, modelClass1._2),
            new DenseMatrix(value_M, value_k, modelClass1._3), new DenseMatrix(value_M, value_D, modelClass1._4))
            .withColumnRenamed("prob", "probMod1").as("valMod1")
            .join(
              hsmm.BaumWelchAlgorithm.validateLength(validClass1, value_M, value_k, value_D, value_T,
                new DenseVector(modelClass0._1), new DenseMatrix(value_M, value_M, modelClass0._2),
                new DenseMatrix(value_M, value_k, modelClass0._3), new DenseMatrix(value_M, value_D, modelClass0._4))
                .withColumnRenamed("prob", "probMod0").as("valMod0"),
              col("valMod1.workitem") === col("valMod0.workitem"), "inner")
            .select(col("valMod1.workitem").as("workitem"), col("probMod1"), col("probMod0"))
        log.info("Saving result validation Class1")
        resultClass1.write.format("com.databricks.spark.csv").save(applicationProps.getProperty("path_sample_Class1_kfold") + inter)
      }

      if (new java.io.File(applicationProps.getProperty("path_sample_Class0_kfold") + inter).exists) {
        log.info("Loading result validation Class0")
        resultClass0 = sparkSession.read.csv(applicationProps.getProperty("path_sample_Class0_kfold") + inter)
          .withColumnRenamed("_c0", "workitem")
          .withColumnRenamed("_c1", "probMod1")
          .withColumnRenamed("_c2", "probMod0")

      }else{
        resultClass0 =
          hsmm.BaumWelchAlgorithm.validateLength(validClass0, value_M, value_k, value_D, value_T,
            new DenseVector(modelClass1._1), new DenseMatrix(value_M, value_M, modelClass1._2),
            new DenseMatrix(value_M, value_k, modelClass1._3), new DenseMatrix(value_M, value_D, modelClass1._4))
            .withColumnRenamed("prob", "probMod1").as("valMod1")
            .join(
              hsmm.BaumWelchAlgorithm.validateLength(validClass0, value_M, value_k, value_D, value_T,
                new DenseVector(modelClass0._1), new DenseMatrix(value_M, value_M, modelClass0._2),
                new DenseMatrix(value_M, value_k, modelClass0._3), new DenseMatrix(value_M, value_D, modelClass0._4))
                .withColumnRenamed("prob", "probMod0").as("valMod0"),
              col("valMod1.workitem") === col("valMod0.workitem"), "inner")
            .select(col("valMod1.workitem").as("workitem"), col("probMod1"), col("probMod0"))
        log.info("Saving result validation Class0")
        resultClass0.write.format("com.databricks.spark.csv").save(applicationProps.getProperty("path_sample_Class0_kfold") + inter)

      }

      /** N value */
      log.info("Compute N")
      val N: Double = validClass1.count + validClass0.count
      log.info("Value of N: " + N)

      /** True Positives */
      log.info("Compute True Positives")
      val TP: Double = resultClass1.where("probMod1 > probMod0").count
      log.info("Value of TP: " + TP)

      /** False Positives */
      log.info("Compute False Positives")
      val FP: Double = resultClass0.where("probMod1 > probMod0").count
      log.info("Value of FP: " + FP)

      /** False Negatives */
      log.info("Compute False Negatives")
      val FN: Double = resultClass1.where("probMod1 <= probMod0").count
      log.info("Value of FN: " + FN)

      /** True Negatives */
      log.info("Compute True Negatives")
      val TN: Double = resultClass0.where("probMod1 <= probMod0").count
      log.info("Value of TN: " + TN)

      /** sensitivity */
      log.info("Compute Sensitivity")
      val sensi: Double = TP / (TP + FN)
      log.info("Value of sensi: " + sensi)

      /** specificity */
      log.info("Compute Specificity")
      val speci: Double = TN / (TN + FP)
      log.info("Value of speci: " + speci)

      /** Accuracy */
      log.info("Compute Accuracy")
      val effic: Double = (TP + TN) / (TP + FP + FN + TN)
      log.info("Value of Accuracy: " + effic)

      /** error */
      log.info("Compute Error")
      val error: Double = 1 - effic
      log.info("Value of error: " + error)

      //trainClass1.unpersist()
      //trainClass0.unpersist()
      //validClass1.unpersist()
      //validClass0.unpersist()
      validClass1.unpersist()
      validClass0.unpersist()

      validClass1 = sparkSession.emptyDataFrame
      validClass0 = sparkSession.emptyDataFrame

      resultClass1 = sparkSession.emptyDataFrame
      resultClass0 = sparkSession.emptyDataFrame

      log.info("*****************************************************************************************")
      hsmm.Utils.writeresult(applicationProps.getProperty("path_result"), N + "," + TP + "," + FP + "," + FN + "," + TN + "," + sensi + "," + speci + "," + effic + "," + error + "\n")
    })
    sparkSession.stop()
  }

  def set_folds(sample: DataFrame, n: Int, kfolds: Int): DataFrame = {
    val randomList = scala.util.Random.shuffle((0 until n).toList)
    val indexList = Array.fill[Int](n)(0)
    (1 until kfolds).foreach(i => (i until n by kfolds).foreach(j => indexList(randomList(j)) = i))
    val udf_setfold: UserDefinedFunction = udf((rowId: Int) => indexList(rowId - 1))
    sample.withColumn("kfold", udf_setfold(col("rowId")))
  }
}