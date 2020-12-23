"""
# -*- coding: utf-8 -*-
@Time    : 12/22/2020 3:36 PM
@Author  : liam
@FileName: embedding.py
@Software: PyCharm
@Describe:

"""
import math
import os
import random

from pyspark.ml.feature import Word2Vec
from pyspark.rdd import PipelinedRDD
from pyspark.sql import SparkSession, DataFrame
import pyspark.sql.functions as F
from pyspark.sql.types import StructType, StructField, StringType, ArrayType, FloatType

from offline.config import Config
from offline.udfs import udf_combine_movies_by_timeline
from offline.utils import load_data, print_info, get_root_path

os.environ['PYSPARK_PYTHON'] = '/home/liam/anaconda3/bin/python'


def process_item_sequence(rating_data):
    sequences = rating_data.filter(F.col("rating") >= 3.5).groupBy("userId")\
        .agg(udf_combine_movies_by_timeline(F.collect_list("movieId"), F.collect_list("timestamp")).alias("movieIds"))\
        .withColumn("movieIdStr", F.array_join(F.col("movieIds"), " "))
    print_info(sequences, message="after build movieIdStr: ")
    sequences = sequences.select("movieIds")
    return sequences


def train_item2vec(spark, movie_sequences, output_path):
    print(movie_sequences.show(10))
    word2vec = Word2Vec().setVectorSize(Config.emb_length).setWindowSize(Config.window_size)\
        .setMaxIter(Config.iter_size).setInputCol("movieIds").setOutputCol("model")
    model = word2vec.fit(movie_sequences)
    synonyms = model.findSynonymsArray("158", 10)
    for synonym, cosineSimilarity in synonyms:
        print(synonym, cosineSimilarity)
    result = model.getVectors().rdd.map(lambda x: [x[0], x[1].toArray().tolist()]).collect()

    with open(output_path, "w+", encoding="utf-8") as f:
        for line in result:
            movie_id = line[0]
            vector = [str(x) for x in line[1]]
            vector = " ".join(vector)
            line = movie_id + ":" + vector + "\n"
            f.write(line)
    return model


def graph_embedding(spark, movie_sequences, output_path):
    pair_data_with_weight, item2weight = generate_transition_matrix(movie_sequences)
    graph_samples = random_walk(spark, pair_data_with_weight, item2weight, Config.sample_count, Config.sample_length)
    train_item2vec(spark, graph_samples, output_path)


def generate_transition_matrix(movie_sequences: DataFrame):
    print_info(movie_sequences)
    pair_data = movie_sequences.rdd.flatMap(build_pair_data)
    print(pair_data.take(20))
    pair_data_with_weight = pair_data.groupBy(lambda x: x[0]).map(build_movieId_weight).collectAsMap()
    item2weight = {}
    total_length = movie_sequences.count()
    for movieId, id2weight in pair_data_with_weight.items():
        item2weight[movieId] = len(id2weight) * 1.0 / total_length
    return pair_data_with_weight, item2weight


def build_pair_data(movieIds):
    pair_data = []
    pre_movieId = None
    for movieId in movieIds[0]:
        if not pre_movieId:
            pre_movieId = movieId
        else:
            pair_data.append([pre_movieId, movieId])
            pre_movieId = movieId
    return pair_data


def build_movieId_weight(line):
    movieId = line[0]
    batch_data = line[1]
    group_length = len(batch_data)
    batch_map = {}
    for line in batch_data:
        pre_movieId, current_movieId = line[0], line[1]
        batch_map[current_movieId] = batch_map.get(current_movieId, 0) + 1
    for current_movieId, weight in batch_map.items():
        batch_map[current_movieId] = batch_map.get(current_movieId) * 1.0 / group_length
    return [movieId, batch_map]


def random_walk(spark: SparkSession, pair_data_with_weight, item2weight, sample_count, sample_length):
    samples = []
    for _ in range(sample_count):
        samples.append([one_random_walk(pair_data_with_weight, item2weight, sample_length)])
    columns = ["movieIds"]
    sample_data_frame = spark.createDataFrame(samples, columns)
    return sample_data_frame


def one_random_walk(pair_data_with_weight, item2weight, sample_length):
    sample = []
    random_prob = random.random()
    acc_prob = 0.0
    # get first movieId
    for movieId, weight in item2weight.items():
        acc_prob += weight
        if random_prob <= acc_prob:
            sample.append(movieId)
            break
    # get movieId after
    for _ in range(sample_length - 1):
        # if it has next neighbor
        if sample[-1] in item2weight.keys():
            random_prob = random.random()
            acc_prob = 0.0
            for other_movieId, weight in pair_data_with_weight[sample[-1]].items():
                acc_prob += weight
                if random_prob <= acc_prob:
                    sample.append(other_movieId)
                    break
    return sample


def generate_user_embedding(spark, rating_data: DataFrame, model, output_path):
    # use user watched movieIds to generate embedding vector
    movieId2vec = model.getVectors().rdd.map(lambda x: [x[0], x[1].toArray().tolist()]).collect()
    struct = StructType(
        [
            StructField("movieId", StringType(), True),
            StructField("embed", ArrayType(FloatType()), True)
        ]
    )
    movieId2vec = spark.createDataFrame(movieId2vec, schema=struct)
    user2vec = rating_data.join(movieId2vec, on=["movieId"], how="left")

def main():
    spark = SparkSession.builder.master("local[1]").appName("embedding").getOrCreate()
    movie_data, rating_data = load_data(spark)
    movie_sequences = process_item_sequence(rating_data)
    # item2vec
    output_path = os.path.join(get_root_path(), "data/models/item2vec.csv")
    model = train_item2vec(spark, movie_sequences, output_path)
    output_path = os.path.join(get_root_path(), "data/models/graph2vec.csv")
    graph_embedding(spark, movie_sequences, output_path)
    output_path = os.path.join(get_root_path(), "data/models/user_embed.csv")
    generate_user_embedding(spark, rating_data, model, output_path)


if __name__ == '__main__':
    main()
