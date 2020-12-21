"""
# -*- coding: utf-8 -*-
@Time    : 12/21/2020 9:24 AM
@Author  : liam
@FileName: feature_eng_for_rec_model.py
@Software: PyCharm
@Describe:

"""
import os

import pyspark.sql as sql
from pyspark.sql import SparkSession, DataFrame
import pyspark.sql.functions as F
from pyspark.sql.types import LongType

from offline import config
from offline.config import Config
from offline.utils import load_data, print_info, udf_get_year_from_title, udf_get_title_from_title, udf_extract_genres, \
    get_root_path

os.environ['PYSPARK_PYTHON'] = '/home/liam/anaconda3/bin/python'


def add_sample_label(data):
    print_info(data)
    sample_count = data.count()
    data.groupBy("rating").count().orderBy("rating")\
        .withColumn("percentage", F.col("count")/sample_count)\
        .show()
    data = data.withColumn("label", F.when(F.col("rating") >= 3.5, 1).otherwise(0))
    print_info(data)
    return data


def add_movie_features(data, rating_with_label):
    # combine movie data and label
    data = rating_with_label.join(data, on=['movieId'], how="left")
    # get release year
    data = data.withColumn("releaseYear", udf_get_year_from_title(F.col("title")))\
        .withColumn("title", udf_get_title_from_title(F.col("title")))\
        .drop("title")
    # split genres to 3 columns
    data = data.withColumn("genre1", F.split(F.col("genres"), "\\|").getItem(0)) \
        .withColumn("genre2", F.split(F.col("genres"), "\\|").getItem(1)) \
        .withColumn("genre3", F.split(F.col("genres"), "\\|").getItem(2))
    # get rating's avg, std
    rating_features = data.groupBy("movieId").agg(
        F.count(F.lit(1)).alias("movieRatingCount"),
        F.format_number(F.avg(F.col("rating")), Config.NUMBER_PRECISION).alias("AvgMovieRating"),
        F.stddev(F.col("rating")).alias("StdMovieRating")).na.fill(0)\
        .withColumn("StdMovieRating", F.format_number(F.col("StdMovieRating"), Config.NUMBER_PRECISION))

    data = data.join(rating_features, on=["movieId"], how="left")
    print_info(data)
    return data


def add_user_features(data):
    # find positive rating list of each userId
    features = data.withColumn("userPositiveHistory",
                               F.collect_list(F.when(F.col("label") == 1, F.col("movieId")).otherwise(F.lit(None)))
                               .over(
                                   sql.Window.partitionBy("userId").orderBy(F.col("timestamp")).rowsBetween(-100, -1)
                               ))\
        .withColumn("userPositiveHistory", F.reverse(F.col("userPositiveHistory"))) \
        .withColumn("userRatedMovie1", F.col("userPositiveHistory").getItem(0)) \
        .withColumn("userRatedMovie2", F.col("userPositiveHistory").getItem(1)) \
        .withColumn("userRatedMovie3", F.col("userPositiveHistory").getItem(2)) \
        .withColumn("userRatedMovie4", F.col("userPositiveHistory").getItem(3)) \
        .withColumn("userRatedMovie5", F.col("userPositiveHistory").getItem(4)) \
        .withColumn("userRatingCount",
                    F.count(F.lit(1)).over(sql.Window.partitionBy("userId")
                    .orderBy(F.col("timestamp")).rowsBetween(-100, -1))) \
        .withColumn("userAvgReleaseYear",
                    F.avg(F.col("releaseYear")).over(sql.Window.partitionBy("userId")
                    .orderBy(F.col("timestamp")).rowsBetween(-100, -1)).cast("integer")) \
        .withColumn("userReleaseYearStddev",
                    F.stddev(F.col("releaseYear")).over(sql.Window.partitionBy("userId")
                    .orderBy(F.col("timestamp")).rowsBetween(-100, -1)).cast("integer")) \
        .withColumn("userAvgRating",
                    F.format_number(F.avg(F.col("rating")).over(sql.Window.partitionBy("userId")
                    .orderBy("timestamp").rowsBetween(-100, -1)), Config.NUMBER_PRECISION)) \
        .withColumn("userRatingStddev",
                    F.format_number(F.stddev(F.col("rating")).over(sql.Window.partitionBy("userId")
                    .orderBy("timestamp").rowsBetween(-100, -1)), Config.NUMBER_PRECISION)) \
        .withColumn("userGenres",
                    udf_extract_genres(F.collect_list(F.when(F.col("label") == 1, F.col("genres")).otherwise(F.lit(None)))
                    .over(sql.Window.partitionBy("userId").orderBy("timestamp").rowsBetween(-100, -1)))) \
        .na.fill(0) \
        .withColumn("userReleaseYearStddev", F.format_number(F.col("userReleaseYearStddev"), Config.NUMBER_PRECISION)) \
        .withColumn("userGenre1", F.col("userGenres").getItem(0)) \
        .withColumn("userGenre2", F.col("userGenres").getItem(1)) \
        .withColumn("userGenre3", F.col("userGenres").getItem(2)) \
        .withColumn("userGenre4", F.col("userGenres").getItem(3)) \
        .withColumn("userGenre5", F.col("userGenres").getItem(4)) \
        .drop("genres", "userGenres", "userPositiveHistory") \
        .filter(F.col("userRatingCount") > 1)
    print_info(features, topN=20)
    return features


def split_and_save_train_test_samples(features: DataFrame, dir_path):
    small_samples = features.sample(0.1)
    train_dataset, test_dataset = small_samples.randomSplit((0.8, 0.2))
    train_save_path = os.path.join(dir_path, "trainSamples")
    test_save_path = os.path.join(dir_path, "testSamples")
    train_dataset.repartition(1).write.options(header=True).mode("overwrite").csv(train_save_path)
    test_dataset.repartition(1).write.options(header=True).mode("overwrite").csv(test_save_path)


def split_and_save_train_test_samples_by_timestamp(features: DataFrame, dir_path):
    small_samples = features.sample(0.1).withColumn("timestampLong", F.col("timestamp").cast(LongType()))
    quantile = small_samples.approxQuantile("timestampLong", [0.8], 0.25)
    split_timestamp = quantile[0]
    train_dataset = small_samples.where(F.col("timestampLong") <= split_timestamp).drop("timestampLong")
    test_dataset = small_samples.where(F.col("timestampLong") > split_timestamp).drop("timestampLong")
    train_save_path = os.path.join(dir_path, "trainSamplesByTimeSplit")
    test_save_path = os.path.join(dir_path, "testSamplesByTimeSplit")
    train_dataset.repartition(1).write.options(header=True).mode("overwrite").csv(train_save_path)
    test_dataset.repartition(1).write.options(header=True).mode("overwrite").csv(test_save_path)


def extract_save_user_features_to_redis(features):
    pass


def extract_and_save_movie_features_to_redis(features: DataFrame):
    samples = features.withColumn("movieRowNum",
                                  F.row_number().over(sql.Window.partitionBy("movieId").orderBy(F.col("timestamp").desc())))\
        .filter(F.col("movieRowNum") == 1)
    print_info(samples)


def main():
    spark = SparkSession.builder.master("local[1]").appName("feature_eng_for_rec_model")\
        .getOrCreate()
    save_dir = os.path.join(get_root_path(), "data/samples")
    movie_data, rating_data = load_data(spark)

    rating_sample_with_label = add_sample_label(rating_data)
    sample_with_movie_features = add_movie_features(movie_data, rating_sample_with_label)
    sample_with_user_features = add_user_features(sample_with_movie_features)

    # split_and_save_train_test_samples(sample_with_user_features, save_dir)
    # split_and_save_train_test_samples_by_timestamp(sample_with_user_features, save_dir)

    extract_and_save_movie_features_to_redis(sample_with_movie_features)
    extract_save_user_features_to_redis(sample_with_user_features)
    spark.stop()


if __name__ == '__main__':
    main()
