"""
# -*- coding: utf-8 -*-
@Time    : 12/18/2020 3:30 PM
@Author  : liam
@FileName: feature_engineering.py
@Software: PyCharm
@Describe:

"""
import os

from pyspark.ml import Pipeline
from pyspark.ml.feature import OneHotEncoder, StringIndexer, QuantileDiscretizer, MinMaxScaler
from pyspark.sql import SparkSession, DataFrame
import pyspark.sql.functions as F

from offline.utils import get_root_path, udf_list2vec, print_info, udf_avg_rating_to_vec

os.environ['PYSPARK_PYTHON'] = '/home/liam/anaconda3/bin/python'


def load_data(spark):
    movie_data_path = os.path.join(get_root_path(), "data/simple/movies.csv")
    rating_data_path = os.path.join(get_root_path(), "data/simple/ratings.csv")
    movie_data = spark.read.options(header=True).csv(movie_data_path)
    rating_data = spark.read.options(header=True).csv(rating_data_path)
    return movie_data, rating_data


def one_hot_example(data: DataFrame):
    data_with_id_number = data.withColumn("movieIdNumber", F.col("movieId").cast("Integer"))
    encoder = OneHotEncoder(inputCol="movieIdNumber", outputCol="movieIdVector")
    model = encoder.fit(data_with_id_number)
    result = model.transform(data_with_id_number)
    print_info(result)


def multi_hot_example(data: DataFrame):
    sample_data_with_genre = data.select(F.col("movieId"), F.explode(F.split(F.col("genres"), "\\|")
                                                                     .cast("array<string>")).alias("genre"))
    indexer = StringIndexer(inputCol="genre", outputCol="genreIndex")
    genre_index = indexer.fit(sample_data_with_genre).transform(sample_data_with_genre)\
        .withColumn("genreIndexInt", F.col("genreIndex").cast("integer"))
    genre_index_size = genre_index.agg(F.max(F.col("genreIndexInt"))).rdd.collect()[0][0] + 1
    genre_data = genre_index.groupBy(F.col("movieId")).agg(F.collect_list("genreIndexInt").alias("genreIndexes"))\
        .withColumn("indexSize", F.lit(genre_index_size))
    final_sample = genre_data.withColumn("genreVector", udf_list2vec(F.col("genreIndexes"), F.col("indexSize")))
    print_info(final_sample)


def numerical_example(data: DataFrame):
    movie_features = data.groupBy("movieId").agg(F.count(F.lit(1)).alias("ratingCount"),
                                                 F.avg("rating").alias("avgRating"),
                                                 F.variance("rating").alias("ratingVar")
                                                 ).withColumn("avgRatingVec", udf_avg_rating_to_vec(F.col("avgRating")))
    print_info(movie_features)
    # bucketing
    rating_count_discretizer = QuantileDiscretizer(numBuckets=100, inputCol="ratingCount", outputCol="ratingCountBucket")
    # normalization
    rating_scaler = MinMaxScaler(inputCol="avgRatingVec", outputCol="scaleAvgRating")
    pipeline_stage = [rating_count_discretizer, rating_scaler]
    feature_pipeline = Pipeline(stages=pipeline_stage)
    movie_processed_features = feature_pipeline.fit(movie_features).transform(movie_features)
    print_info(movie_processed_features)


def main():
    # init sparkSession
    spark = SparkSession.builder.master("local[1]").appName("feature_engineering").getOrCreate()
    print("load data")
    movie_data, rating_data = load_data(spark)
    print("Movie data Sample:")
    print_info(movie_data)
    print("Rating data Sample:")
    print_info(rating_data)

    print("one-hot encoder example on spark")
    one_hot_example(movie_data)
    print("multi-hot encoder example on spark")
    multi_hot_example(movie_data)
    print("numerical features example")
    numerical_example(rating_data)


if __name__ == '__main__':
    main()
