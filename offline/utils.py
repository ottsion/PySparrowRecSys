"""
# -*- coding: utf-8 -*-
@Time    : 12/18/2020 3:39 PM
@Author  : liam
@FileName: utils.py
@Software: PyCharm
@Describe:

"""
import os

from pyspark.sql import DataFrame


def get_root_path():
    path = os.path.abspath(os.path.dirname(__file__))
    root_dir = path.split("offline")[0]
    return root_dir


def load_data(spark):
    movie_data_path = os.path.join(get_root_path(), "data/simple/movies.csv")
    rating_data_path = os.path.join(get_root_path(), "data/simple/ratings.csv")
    movie_data = spark.read.options(header=True).csv(movie_data_path)
    rating_data = spark.read.options(header=True).csv(rating_data_path)
    return movie_data, rating_data


def print_info(df: DataFrame, topN=5, message=""):
    print("---------------------------")
    if message:
        print("**-> %s" % message)
    df.printSchema()
    df.show(topN)
    print("---------------------------")


