"""
# -*- coding: utf-8 -*-
@Time    : 12/18/2020 3:39 PM
@Author  : liam
@FileName: utils.py
@Software: PyCharm
@Describe:

"""
import os

from pyspark.ml import linalg
import pyspark.sql.functions as F
from pyspark.ml.linalg import VectorUDT, Vectors
from pyspark.sql import DataFrame


def get_root_path():
    path = os.path.abspath(os.path.dirname(__file__))
    root_dir = path.split("offline")[0]
    return root_dir


def print_info(df: DataFrame):
    print("---------------------------")
    df.printSchema()
    df.show(5)
    print("---------------------------")


def list2vec(indexes, size):
    indexes = sorted(indexes)
    values = [1.0 for _ in range(len(indexes))]
    output = linalg.Vectors.sparse(size, indexes, values)
    return output


def avg_rating_to_vec(data):
    return Vectors.dense(data)


udf_list2vec = F.udf(list2vec, VectorUDT())
udf_avg_rating_to_vec = F.udf(avg_rating_to_vec, VectorUDT())
