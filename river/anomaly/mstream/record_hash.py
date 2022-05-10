import math
import random
from itertools import compress
import numpy as np


def counts_to_anom(tot, cur, cur_t):
    cur_mean = tot / cur_t
    sqerr = pow(max(0, cur - cur_mean), 2)
    return sqerr / cur_mean + sqerr / (cur_mean * max(1, cur_t - 1))


class RecordHash():
    def __init__(self, is_categorical, num_buckets, number_hash_functions):
        self.is_categorical = is_categorical
        self.is_numerical = [not feature for feature in is_categorical]
        # self.num_rows = num_rows number of hash functions, we will just use 2 for now
        self.num_buckets = num_buckets
        self.number_numerical_features = len(self.is_numerical)
        self.number_categorical_features = len(self.is_categorical)
        self.number_hash_functions = number_hash_functions  # num_rows in original code
        self.numerical_hash = RecordNumericalHash(
            self.number_hash_functions, self.num_buckets, self.number_numerical_features)
        self.categorical_hash = RecordCategorialHash(
            self.number_hash_functions, self.num_buckets, self.number_categorical_features)
        self.clear()

    def insert(self, x):
        # divide x into categorical and numerical
        x_categorical = compress(x, self.is_categorical)
        x_numerical = compress(x, self.is_numerical)

        for i in range(self.number_hash_functions):
            bucket_categorical = self.categorical_hash.hash(x_categorical, i)
            bucket_numerical = self.numerical_hash.hash(x_numerical, i)

            bucket = (bucket_categorical + bucket_numerical) % self.num_buckets
            self.count[i][bucket] += 1
            self.total_count[i][bucket] += 1

    def get_count(self, x, t):
        # TODO: Extract duplicate code from insert and get_count
        x_categorical = compress(x, self.is_categorical)
        x_numerical = compress(x, self.is_numerical)

        min_count = float('inf')  # init with max value
        min_total_count = float('inf')

        for i in range(self.number_hash_functions):
            bucket_categorical = self.categorical_hash.hash(x_categorical, i)
            bucket_numerical = self.numerical_hash.hash(x_numerical, i)
            bucket = (bucket_categorical + bucket_numerical) % self.num_buckets

            min_count = min(min_count, self.count[i][bucket])
            min_total_count = min(min_total_count, self.total_count[i][bucket])

        return counts_to_anom(min_total_count, min_count, t)

    def clear(self):
        self.count = [[0 for _ in range(self.num_buckets)]
                      for _ in range(self.number_hash_functions)]
        self.total_count = [[0 for _ in range(self.num_buckets)]
                            for _ in range(self.number_hash_functions)]

    def lower(self, factor):
        for i in range(self.number_hash_functions):
            for j in range(self.num_buckets):
                self.count[i][j] *= factor


class RecordCategorialHash():
    def __init__(self, number_hash_functions, num_buckets, number_categorical_features):
        self.cat_recordhash = []
        self.number_categorical_features = number_categorical_features
        self.num_buckets = num_buckets
        self.number_hash_functions = number_hash_functions
        for i in range(self.number_categorical_features):
            self.cat_recordhash.append({})
        self.clear()

    def hash(self, x_categorical, i):
        resid = 0
        for i in range(self.number_categorical_features):
            current_category = self.cat_recordhash[i]
            key = hash(x_categorical[i])
            if current_category[key]:
                current_category[key] += 1
            else:
                current_category[key] = 1
            value = current_category[key]
            resid = (resid + key * value) % self.num_buckets
        self.bucket_cat = resid + (self.num_buckets if resid < 0 else 0)

    def clear(self):
        self.bucket_cat = 0
        self.categorial_count = [[0 for _ in range(self.num_buckets)]
                                 for _ in range(self.number_features)]

    def get_count(self):
        return self.bucket_cat


class RecordNumericalHash():
    def __init__(self, number_hash_functions, num_buckets, number_numerical_features):
        self.number_numerical_features = number_numerical_features
        self.num_buckets = num_buckets
        self.number_hash_functions = number_hash_functions  # num_rows in original code
        self.k = math.ceil(math.log2(self.num_buckets))
        self.rand_vectors = np.random.normal(0, 1, size=(self.k, number_numerical_features))
        self.clear()

    def hash(self, x_numerical, i):
        x_numerical_vector = np.array(x_numerical)
        bitset = ""
        for i in range(self.k):
            dot_product = np.dot(x_numerical_vector, self.rand_vectors[i].T)
            if dot_product > 0:
                bitset += "1"
            else:
                bitset += "0"
        self.bucket_num = int(bitset, 2)

    def clear(self):
        self.bucket_num = 0
        self.numerical_count = [[[0 for _ in range(self.number_buckets)] for _ in range(
            self.k)] for _ in range(self.number_hash_functions)]

    def get_count(self):
        return self.bucket_num
