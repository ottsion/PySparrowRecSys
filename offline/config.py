"""
# -*- coding: utf-8 -*-
@Time    : 12/21/2020 10:27 AM
@Author  : liam
@FileName: config.py
@Software: PyCharm
@Describe:

"""


class Config(object):
    NUMBER_PRECISION = 2

    user_feature_prefix = "uf:"
    movie_feature_prefix = "mf:"

    emb_length = 10
    window_size = 5
    iter_size = 10
    redis_key_prefix = "i2vEmb"

    sample_count = 20000
    sample_length = 10


