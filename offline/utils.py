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
from pyspark.sql.types import IntegerType, StringType, ArrayType


def get_root_path():
    path = os.path.abspath(os.path.dirname(__file__))
    root_dir = "file:///" + path.split("offline")[0]
    return root_dir


def load_data(spark):
    movie_data_path = os.path.join(get_root_path(), "data/simple/movies.csv")
    rating_data_path = os.path.join(get_root_path(), "data/simple/ratings.csv")
    movie_data = spark.read.options(header=True).csv(movie_data_path)
    rating_data = spark.read.options(header=True).csv(rating_data_path)
    return movie_data, rating_data


def print_info(df: DataFrame, topN=5):
    print("---------------------------")
    df.printSchema()
    df.show(topN)
    print("---------------------------")


def list2vec(indexes, size):
    indexes = sorted(indexes)
    values = [1.0 for _ in range(len(indexes))]
    output = linalg.Vectors.sparse(size, indexes, values)
    return output


def avg_rating_to_vec(data):
    return Vectors.dense(data)


def get_year_from_title(data: str):
    if len(data) < 5:
        return 1990
    year = data.strip()[-5: -1]
    return int(year)


def get_title_from_title(data: str):
    if len(data) < 5:
        return data
    title = data.strip()[: -6]
    return title


def get_extract_genres(genres_list):
    """
    pass in a list which format like ["Action|Adventure|Sci-Fi|Thriller", "Crime|Horror|Thriller"]
    count by each genre，return genre_list in reverse order
    eg:
    (('Thriller',2),('Action',1),('Sci-Fi',1),('Horror', 1), ('Adventure',1),('Crime',1))
    return:['Thriller','Action','Sci-Fi','Horror','Adventure','Crime']
    :param data:
    :return:
    """
    data = {}
    for genres in genres_list:
        arr = genres.split("|")
        for genre in arr:
            data[genre] = data.get(genre, 0) + 1
    sorted_data = sorted(data.items(), key=lambda x: x[1], reverse=True)
    result = [genre[0] for genre in sorted_data]
    return result


udf_list2vec = F.udf(list2vec, VectorUDT())
udf_avg_rating_to_vec = F.udf(avg_rating_to_vec, VectorUDT())
udf_get_year_from_title = F.udf(get_year_from_title, IntegerType())
udf_get_title_from_title = F.udf(get_title_from_title, StringType())
udf_extract_genres = F.udf(get_extract_genres, ArrayType(StringType()))
