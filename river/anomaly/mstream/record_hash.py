import math
import random
from itertools import compress
import numpy as np

class RecordHash():
    def __init__(self, is_categorical, number_buckets, dimension_1, dimension_2, seed):
        self.is_categorical = is_categorical
        self.is_numerical = [ not feature for feature in is_categorical ]
        # self.num_rows = num_rows number of hash functions, we will just use 2 for now
        self.number_buckets = number_buckets
        self.categorical_hash = RecordCategorialHash()
        self.numerical_hash = RecordNumericalHash()
        self.random_num = math.r
        self.dimension_1 = dimension_1 # number of numerical features
        self.dimension_2 = dimension_2 # number of categorical features
        random.seed(seed)


    def insert(self, x):
        # divide x into categorical and numerical
        x_categorical = compress(x, self.is_categorical)
        x_numerical = compress(x, self.is_numerical)

        # create categorical
        bucket_cat = self.categorical_hash.hash(x_categorical)
        bucket_num = self.numerical_hash.hash()

        return (bucket_cat + bucket_num) % self.number_buckets


class RecordCategorialHash():
    def __init__(self, dimension_2, num_buckets):
        self.cat_recordhash = []
        self.dimension_2 = dimension_2
        self.num_buckets = num_buckets
        for i in range(0, self.dimension_2):
            self.cat_recordhash.append({})

    def hash(self, x_categorical):
        resid = 0
        for i in range(0, self.dimension_2):
            current_category = self.cat_recordhash[i]
            key = hash(x_categorical[i])
            if current_category[key]:
                current_category[key] += 1
            else:
                current_category[key] = 1
            value = current_category[key]
            resid = (resid + key * value) % self.num_buckets
        return resid + (0 if resid < 0 else self.num_buckets)

class RecordNumericalHash():
    def __init__(self, dimension_1, num_buckets):
        self.dimension_1 = dimension_1
        self.num_buckets = num_buckets
        self.k = math.ceil(math.log2(self.num_buckets))
        p = dimension_1
        self.rand_vectors = np.random.normal(0, 1, size=(self.k, p))

    def hash(self, x_numerical):
        x_numerical_vector = np.array(x_numerical)
        bitset = ""
        for i in range(0, self.k):
            dot_product = np.dot(x_numerical_vector, self.rand_vectors[i].T)
            if dot_product > 0:
                bitset += "1"
            else:
                bitset += "0"
        return int(bitset, 2)