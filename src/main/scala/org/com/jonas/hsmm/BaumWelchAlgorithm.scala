package org.com.jonas.hsmm

import scala.util.control.Breaks._
import breeze.linalg.{DenseMatrix, DenseVector, normalize, sum}
import org.apache.spark.sql.Row
import org.apache.spark.sql.functions._
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.expressions.UserDefinedFunction
import org.com.jonas.hsmm

object BaumWelchAlgorithm {

  /** * function with reduce function ****/
  def run1(observations: DataFrame, M: Int, k: Int, D: Int,
           initialPi: DenseVector[Double], initialA: DenseMatrix[Double], initialB: DenseMatrix[Double], initialP: DenseMatrix[Double],
           numPartitions: Int = 1, epsilon: Double = 0.0001, maxIterations: Int = 10000,
           kfold: Int, path_Class_baumwelch: String):
  (DenseVector[Double], DenseMatrix[Double], DenseMatrix[Double], DenseMatrix[Double]) = {

    var prior = initialPi
    var transmat = initialA
    var obsmat = initialB
    var durmat = initialP
    var antloglik: Double = Double.NegativeInfinity
    val log = org.apache.log4j.LogManager.getRootLogger

    var inInter = 0
    if (new java.io.File(path_Class_baumwelch + kfold).exists) {
      inInter = scala.io.Source.fromFile(path_Class_baumwelch + kfold).getLines.size - 1
      val stringModel: List[String] = scala.io.Source.fromFile(path_Class_baumwelch + kfold).getLines().toList
      val arraymodel = stringModel.last.split(";")
      prior = new DenseVector(arraymodel(5).split(",").map(_.toDouble))
      transmat = new DenseMatrix(M, M, arraymodel(6).split(",").map(_.toDouble))
      obsmat = new DenseMatrix(M, k, arraymodel(7).split(",").map(_.toDouble))
      durmat = new DenseMatrix(M, D, arraymodel(8).split(",").map(_.toDouble))
      antloglik = arraymodel(9).toDouble
    } else hsmm.Utils.writeresult(path_Class_baumwelch + kfold, "kfold;iteration;M;k;Pi;A;B;P;loglik\n")

    observations.persist()
    var obstrained = observations
      .withColumn("M", lit(M))
      .withColumn("k", lit(k))
      .withColumn("D", lit(D))
      .withColumn("Pi", lit(initialPi.toArray))
      .withColumn("A", lit(initialA.toArray))
      .withColumn("B", lit(initialB.toArray))
      .withColumn("P", lit(initialP.toArray))
      .withColumn("obs", udf_toarray(col("str_obs")))
      .withColumn("T", udf_obssize(col("obs")))

    breakable {
      (inInter until maxIterations).foreach(it => {
        log.info("-----------------------------------------------------------------------------------------")
        log.info("Start Iteration: " + it)
        val newvalues = obstrained.repartition(numPartitions)
          .withColumn("obslik", udf_multinomialprob(col("obs"), col("M"), col("k"), col("T"), col("B")))
          .withColumn("fwdback", udf_fwdback(col("M"), col("k"), col("D"), col("T"), col("Pi"), col("A"), col("P"), col("obslik"), col("obs")))
          .withColumn("loglik", udf_loglik(col("fwdback")))
          .withColumn("prior", udf_newPi(col("fwdback")))
          .withColumn("transmat", udf_newA(col("fwdback")))
          .withColumn("obsmat", udf_newB(col("fwdback")))
          .withColumn("durmat", udf_newP(col("fwdback")))
          .drop("workitem", "str_obs", "M", "k", "D", "Pi", "A", "B", "P", "obs", "T", "obslik", "fwdback")
          .reduce((row1, row2) =>
            Row(row1.getAs[Double](0) + row2.getAs[Double](0),
              (row1.getAs[Seq[Double]](1), row2.getAs[Seq[Double]](1)).zipped.map(_ + _),
              (row1.getAs[Seq[Double]](2), row2.getAs[Seq[Double]](2)).zipped.map(_ + _),
              (row1.getAs[Seq[Double]](3), row2.getAs[Seq[Double]](3)).zipped.map(_ + _),
              (row1.getAs[Seq[Double]](4), row2.getAs[Seq[Double]](4)).zipped.map(_ + _)))

        val loglik = newvalues.getAs[Double](0)
        log.info("LogLikehood Value: " + loglik)
        /*
        prior = normalize(new DenseVector(newvalues.getAs[Seq[Double]](1).toArray) :+= Math.pow(2, -52), 1.0)
        transmat = Utils.mkstochastic(new DenseMatrix(M, M, newvalues.getAs[Seq[Double]](2).toArray) :+= Math.pow(2, -52))
        obsmat = Utils.mkstochastic(new DenseMatrix(M, k, newvalues.getAs[Seq[Double]](3).toArray) :+= Math.pow(2, -52))
        durmat = Utils.normalise(new DenseMatrix(M, D, newvalues.getAs[Seq[Double]](4).toArray) :+= Math.pow(2, -52))
        */
        prior = normalize(new DenseVector(newvalues.getAs[Seq[Double]](1).toArray), 1.0)
        transmat = Utils.mkstochastic(new DenseMatrix(M, M, newvalues.getAs[Seq[Double]](2).toArray))
        obsmat = Utils.mkstochastic(new DenseMatrix(M, k, newvalues.getAs[Seq[Double]](3).toArray))
        //durmat = Utils.mkstochastic(new DenseMatrix(M, D, newvalues.getAs[Seq[Double]](4).toArray))
        durmat = Utils.normalise(new DenseMatrix(M, D, newvalues.getAs[Seq[Double]](4).toArray))

        hsmm.Utils.writeresult(path_Class_baumwelch + kfold,
          kfold + ";" +
            it + ";" +
            M + ";" +
            k + ";" +
            D + ";" +
            prior.toArray.mkString(",") + ";" +
            transmat.toArray.mkString(",") + ";" +
            obsmat.toArray.mkString(",") + ";" +
            durmat.toArray.mkString(",") + ";" +
            loglik + "\n")

        if (Utils.emconverged(loglik, antloglik, epsilon)) {
          log.info("End Iteration: " + it)
          log.info("-----------------------------------------------------------------------------------------")
          break
        }
        antloglik = loglik

        obstrained.unpersist()
        obstrained = observations
          .withColumn("M", lit(M))
          .withColumn("k", lit(k))
          .withColumn("D", lit(D))
          .withColumn("Pi", lit(prior.toArray))
          .withColumn("A", lit(transmat.toArray))
          .withColumn("B", lit(obsmat.toArray))
          .withColumn("P", lit(initialP.toArray))
          .withColumn("obs", udf_toarray(col("str_obs")))
          .withColumn("T", udf_obssize(col("obs")))
        log.info("End Iteration: " + it)
        log.info("-----------------------------------------------------------------------------------------")
      })
    }
    (prior, transmat, obsmat, durmat)
  }

  def validate(observations: DataFrame, M: Int, k: Int, D: Int,
               initialPi: DenseVector[Double], initialA: DenseMatrix[Double], initialB: DenseMatrix[Double], initialP: DenseMatrix[Double]):
  DataFrame = {
    observations
      .withColumn("M", lit(M))
      .withColumn("k", lit(k))
      .withColumn("D", lit(D))
      .withColumn("Pi", lit(initialPi.toArray))
      .withColumn("A", lit(initialA.toArray))
      .withColumn("B", lit(initialB.toArray))
      .withColumn("P", lit(initialP.toArray))
      .withColumn("obs", udf_toarray(col("str_obs")))
      .withColumn("T", udf_obssize(col("obs")))
      .withColumn("obslik", udf_multinomialprob(col("obs"), col("M"), col("k"), col("T"), col("B")))
      .withColumn("prob", udf_fwd(col("M"), col("D"), col("T"), col("Pi"), col("A"), col("P"), col("obslik")))
      .drop("str_obs", "M", "k", "D", "Pi", "A", "B", "P", "obs", "T", "obslik")
  }

  /** * udf functions ****/
  val udf_toarray: UserDefinedFunction = udf((s: String) => s.split(";").map(_.toInt))
  val udf_obssize: UserDefinedFunction = udf((s: Seq[Int]) => s.length)

  /** * udf_multinomialprob ****/
  val udf_multinomialprob: UserDefinedFunction = udf((obs: Seq[Int], M: Int, k: Int, T: Int, B: Seq[Double]) => {
    val funB: DenseMatrix[Double] = new DenseMatrix(M, k, B.toArray)
    val output: DenseMatrix[Double] = DenseMatrix.tabulate(M, T) { case (m, t) => funB(m, obs(t)) }
    output.toArray
  })

  val udf_fwdback: UserDefinedFunction = udf((M: Int, k: Int, D: Int, T: Int, Pi: Seq[Double], A: Seq[Double], P: Seq[Double], obslik: Seq[Double], obs: Seq[Int]) => {

    val funPi: DenseVector[Double] = new DenseVector(Pi.toArray)
    val funA: DenseMatrix[Double] = new DenseMatrix(M, M, A.toArray)
    val funP: DenseMatrix[Double] = new DenseMatrix(M, D, P.toArray)
    val funObslik: DenseMatrix[Double] = new DenseMatrix(M, T, obslik.toArray)

    /**
      * Matriz u(t,j,d), (5.9)
      */
    val matrixu: DenseVector[DenseMatrix[Double]] = DenseVector.fill(T) {
      DenseMatrix.zeros[Double](M, D)
    }

    /** * valores iniciales **/
    (0 until T).foreach(t =>
      (0 until M).foreach(j => matrixu(t)(j, 0) = funObslik(j, t)))

    (0 until T).foreach(t =>
      (0 until M).foreach(j =>
        (1 until D).foreach(d =>
          if (d - 1 > -1 && t - d + 1 > -1)
            matrixu(t)(j, d) = matrixu(t)(j, d - 1) * funObslik(j, t - d + 1))))

    /**
      * Forwards variables
      */
    val scale: DenseVector[Double] = DenseVector.ones[Double](T)
    val alpha: DenseMatrix[Double] = DenseMatrix.zeros[Double](M, T)
    val alphaprime: DenseMatrix[Double] = DenseMatrix.zeros[Double](M, T + 1)

    alphaprime(::, 0) := normalize(funPi, 1.0)

    (0 until T).foreach(t => {
      (0 until M).foreach(j =>
        (0 until D).foreach(d =>
          if (t - d + 1 > -1 && t - d + 1 < T + 1)
            alpha(j, t) = alpha(j, t) + (alphaprime(j, t - d + 1) * funP(j, d) * matrixu(t)(j, d))))

      alpha(::, t) := Utils.normalise(alpha(::, t), scale, t)

      (0 until M).foreach(j =>
        (0 until M).foreach(i => if (i != j) alphaprime(j, t + 1) = alphaprime(j, t + 1) + alpha(i, t) * funA(i, j)))

      alphaprime(::, t + 1) := normalize(alphaprime(::, t + 1), 1.0)
    })

    //var loglik: Double = 0.0
    //if (scale.toArray.filter(i => i == 0).isEmpty) loglik = sum(scale.map(Math.log)) else loglik = Double.NegativeInfinity
    //if (scale.findAll(i => i == 0).isEmpty) loglik = sum(scale.map(Math.log)) else loglik = Double.NegativeInfinity
    //Modificación validar si funciona
    val loglik: Double = sum(scale.map(Math.log))

    /**
      * Backwards variables
      * el beta en la posición cero es cero, why?
      */
    val beta: DenseMatrix[Double] = DenseMatrix.zeros[Double](M, T)
    val betaprime: DenseMatrix[Double] = DenseMatrix.zeros[Double](M, T + 1)

    beta(::, T - 1) := 1.0

    val t = T - 1
    (0 until M).foreach(j =>
      (0 until D).foreach(d =>
        if (t + d < T)
          betaprime(j, t + 1) = betaprime(j, t + 1) + funP(j, d) * matrixu(t + d)(j, d) * beta(j, t + d)))

    betaprime(::, t + 1) := normalize(betaprime(::, t + 1), 1.0)

    /** probar con (t <- T - 2 to -1 by -1) **/
    for (t <- T - 2 to -1 by -1) {
      (0 until M).foreach(j =>
        (0 until D).foreach(d =>
          if (t + d < T)
            betaprime(j, t + 1) = betaprime(j, t + 1) + funP(j, d) * matrixu(t + d)(j, d) * beta(j, t + d)))

      betaprime(::, t + 1) := normalize(betaprime(::, t + 1), 1.0)

      (0 until M).foreach(j =>
        (0 until M).foreach(i => if (i != j) beta(j, t) = beta(j, t) + funA(j, i) * betaprime(i, t + 1)))

      beta(::, t) := normalize(beta(::, t), 1.0)
    }

    /**
      * Matriz n(t,i,d)
      */
    val matrixn: DenseVector[DenseMatrix[Double]] = DenseVector.fill(T) {
      DenseMatrix.zeros[Double](M, D)
    }
    (0 until T).foreach(t =>
      (0 until M).foreach(i => {
        (0 until D).foreach(d =>
          if (t - d + 1 > -1 && t - d + 1 < T + 1)
            matrixn(t)(i, d) = alphaprime(i, t - d + 1) * funP(i, d) * matrixu(t)(i, d) * beta(i, t))
        matrixn(t)(i, ::) := normalize(matrixn(t)(i, ::).t, 1.0).t
      }))

    /**
      * Matriz xi(t,i,j)
      */
    val matrixi: DenseVector[DenseMatrix[Double]] = DenseVector.fill(T) {
      DenseMatrix.zeros[Double](M, M)
    }
    (0 until T).foreach(t => {
      (0 until M).foreach(i =>
        (0 until M).foreach(j => matrixi(t)(i, j) = alpha(i, t) * funA(i, j) * betaprime(j, t + 1)))
      matrixi(t) = Utils.mkstochastic(matrixi(t))
      //matrixi(t) = Utils.normalise(matrixi(t))
    })

    /**
      * Matriz gamma(t, i)
      */
    val matrixg: DenseMatrix[Double] = DenseMatrix.zeros[Double](M, T)

    (0 until M).foreach(i => matrixg(i, 0) = funPi(i) * betaprime(i, 0))
    matrixg(::, 0) := normalize(matrixg(::, 0), 1.0)

    (1 until T - 1).foreach(t => {
      (0 until M).foreach(i => matrixg(i, t) = matrixg(i, t - 1) + alphaprime(i, t) * betaprime(i, t) - alpha(i, t - 1) * beta(i, t - 1))
      matrixg(::, t) := normalize(matrixg(::, t), 1.0)
    })

    (0 until M).foreach(i => matrixg(i, T - 1) = alpha(i, T - 1))
    matrixg(::, T - 1) := normalize(matrixg(::, T - 1), 1.0)

    /**
      * Matriz newA, estimation of a(i,j)
      */
    val newA = DenseMatrix.zeros[Double](M, M)
    (0 until M).foreach(i => {
      (0 until M).foreach(j => {
        var num = 0.0
        (0 until T).foreach(t => num = num + matrixi(t)(i, j))
        var den = 0.0
        (0 until M).foreach(j2 =>
          (0 until T).foreach(t => den = den + matrixi(t)(i, j2)))
        newA(i, j) = num / den
      })
      //newA(i, ::) := normalize(newA(i, ::).t :+= Math.pow(2, -52), 1.0).t
      newA(i, ::) := normalize(newA(i, ::).t, 1.0).t
    })

    /**
      * Matriz newB, estimation of b(i,vk)
      */
    val newB = DenseMatrix.zeros[Double](M, k)
    (0 until M).foreach(i => {
      (0 until k).foreach(v => {
        var num = 0.0
        (0 until T).foreach(t =>
          if (obs(t) == v)
            num = num + matrixg(i, t))
        var den = 0.0
        (0 until T).foreach(t => den = den + matrixg(i, t))
        newB(i, v) = num / den
      })
      //newB(i, ::) := normalize(newB(i, ::).t :+= Math.pow(2, -52), 1.0).t
      newB(i, ::) := normalize(newB(i, ::).t, 1.0).t
    })

    /**
      * Matriz newP, estimation of p(i,d)
      */
    val newP = DenseMatrix.zeros[Double](M, D)
    (0 until M).foreach(i => {
      (0 until D).foreach(d => {
        var num = 0.0
        (0 until T).foreach(t => num = num + matrixn(t)(i, d))
        var den = 0.0
        (0 until D).foreach(d2 => (0 until T).foreach(t => den = den + matrixn(t)(i, d2)))
        newP(i, d) = num / den
      })
      //newP := Utils.normalise(newP :+= Math.pow(2, -52))
      //newP(i, ::) := normalize(newP(i, ::).t, 1.0).t
    })
    newP := Utils.normalise(newP)

    /**
      * Matriz newPi, estimation of pi(i)
      */
    val newPi = DenseVector.zeros[Double](M)
    (0 until M).foreach(i => {
      var den = 0.0
      (0 until M).foreach(j => den = den + matrixg(j, 0))
      newPi(i) = matrixg(i, 0) / den
    })
    //newPi := normalize(newPi :+= Math.pow(2, -52), 1.0)
    newPi := normalize(newPi, 1.0)

    (loglik, newPi.toArray, newA.toArray, newB.toArray, newP.toArray)
  })

  val udf_loglik: UserDefinedFunction = udf((input: Row) => input.get(0).asInstanceOf[Double])
  val udf_newPi: UserDefinedFunction = udf((input: Row) => input.get(1).asInstanceOf[Seq[Double]])
  val udf_newA: UserDefinedFunction = udf((input: Row) => input.get(2).asInstanceOf[Seq[Double]])
  val udf_newB: UserDefinedFunction = udf((input: Row) => input.get(3).asInstanceOf[Seq[Double]])
  val udf_newP: UserDefinedFunction = udf((input: Row) => input.get(4).asInstanceOf[Seq[Double]])

  /** * Por optimizar ****/
  val udf_fwd: UserDefinedFunction = udf((M: Int, D: Int, T: Int, Pi: Seq[Double], A: Seq[Double], P: Seq[Double], obslik: Seq[Double]) => {

    val funPi: DenseVector[Double] = new DenseVector(Pi.toArray)
    val funA: DenseMatrix[Double] = new DenseMatrix(M, M, A.toArray)
    val funP: DenseMatrix[Double] = new DenseMatrix(M, D, P.toArray)
    val funObslik: DenseMatrix[Double] = new DenseMatrix(M, T, obslik.toArray)

    /**
      * Matriz u(t,j,d), (5.9)
      */
    val matrixu: DenseVector[DenseMatrix[Double]] = DenseVector.fill(T) {
      DenseMatrix.zeros[Double](M, D)
    }

    /** * valores iniciales **/
    (0 until T).foreach(t =>
      (0 until M).foreach(j => matrixu(t)(j, 0) = funObslik(j, t)))

    (0 until T).foreach(t =>
      (0 until M).foreach(j =>
        (1 until D).foreach(d =>
          if (d - 1 > -1 && t - d + 1 > -1)
            matrixu(t)(j, d) = matrixu(t)(j, d - 1) * funObslik(j, t - d + 1))))

    /**
      * Forwards variables
      */
    val scale: DenseVector[Double] = DenseVector.ones[Double](T)
    val alpha: DenseMatrix[Double] = DenseMatrix.zeros[Double](M, T)
    val alphaprime: DenseMatrix[Double] = DenseMatrix.zeros[Double](M, T + 1)

    alphaprime(::, 0) := normalize(funPi, 1.0)

    (0 until T).foreach(t => {
      (0 until M).foreach(j =>
        (0 until D).foreach(d =>
          if (t - d + 1 > -1 && t - d + 1 < T + 1)
            alpha(j, t) = alpha(j, t) + (alphaprime(j, t - d + 1) * funP(j, d) * matrixu(t)(j, d))))

      alpha(::, t) := Utils.normalise(alpha(::, t), scale, t)

      (0 until M).foreach(j =>
        (0 until M).foreach(i => if (i != j) alphaprime(j, t + 1) = alphaprime(j, t + 1) + alpha(i, t) * funA(i, j)))

      alphaprime(::, t + 1) := normalize(alphaprime(::, t + 1), 1.0)
    })

    //var loglik: Double = 0.0
    //if (scale.toArray.filter(i => i == 0).isEmpty) loglik = sum(scale.map(Math.log)) else loglik = Double.NegativeInfinity
    //if (scale.findAll(i => i == 0).isEmpty) loglik = sum(scale.map(Math.log)) else loglik = Double.NegativeInfinity
    //Modificación validar si funciona
    val loglik: Double = sum(scale.map(Math.log))
    loglik
  })


}