package org.com.jonas.nhsmm

import breeze.linalg.{DenseMatrix, DenseVector, Transpose, normalize, sum}

object Utils {

  def emconverged(loglik: Double, previous_loglik: Double, threshold: Double): Boolean = {
    val eps = Math.pow(2, -52)
    val delta_loglik = Math.abs(loglik - previous_loglik)
    val avg_loglik = (Math.abs(loglik) + Math.abs(previous_loglik) + eps) / 2
    if ((delta_loglik / avg_loglik) < threshold) true else false
  }

  //def normalise(input: DenseVector[Double]): DenseVector[Double] = input :*= 1 / sum(input)

  def normalise(input: DenseMatrix[Double]): DenseMatrix[Double] = input :*= 1 / sum(input)

  def normalise(input: DenseVector[Double], inscale: DenseVector[Double], index: Int): DenseVector[Double] = {
    val rsum = sum(input)
    inscale(index) = rsum
    if (rsum == 0) input else input :*= 1 / rsum
  }

  def mkstochastic(input: DenseMatrix[Double]): DenseMatrix[Double] = {
    (0 until input.rows).foreach(i => input(i, ::) := normalize(input(i, ::).t, 1.0).t)
    input
  }

  def writeresult(path: String, result: String): Unit = {
    import java.io._
    val fw = new FileWriter(path, true)
    fw.write(result)
    fw.close()
  }

}
