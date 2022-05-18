import math
import numpy as np
import random
from datetime import datetime
from .utils import counts_to_anom


class RecordHash():
    def __init__(self, number_buckets, number_hash_functions, feature_types):
        self.number_buckets = number_buckets
        number_numerical_features = feature_types.count(False)
        number_categorical_features = feature_types.count(True)
        self.number_hash_functions = number_hash_functions  # num_rows in original code
        self.numerical_hash = RecordNumericalHash(
            self.number_hash_functions, self.number_buckets, number_numerical_features)
        self.categorical_hash = RecordCategorialHash(
            self.number_hash_functions, self.number_buckets, number_categorical_features)
        self.clear()

    def insert(self, x_numerical, x_categorical):
        for i in range(self.number_hash_functions):
            bucket_numerical = self.numerical_hash.hash(x_numerical, i)
            bucket_categorical = self.categorical_hash.hash(x_categorical, i)
            bucket = (bucket_categorical + bucket_numerical) % self.number_buckets
            self.count[i][bucket] += 1
            self.total_count[i][bucket] += 1

    def get_count(self, x_numerical, x_categorical, timestamp):
        min_count = float('inf')  # init with max value
        min_total_count = float('inf')

        for i in range(self.number_hash_functions):
            bucket_numerical = self.numerical_hash.hash(x_numerical, i)
            bucket_categorical = self.categorical_hash.hash(x_categorical, i)
            bucket = (bucket_numerical + bucket_categorical) % self.number_buckets

            min_count = min(min_count, self.count[i][bucket])
            min_total_count = min(min_total_count, self.total_count[i][bucket])
        return counts_to_anom(min_total_count, min_count, timestamp)

    def clear(self):
        self.count = [[0 for _ in range(self.number_buckets)]
                      for _ in range(self.number_hash_functions)]
        self.total_count = [[0 for _ in range(self.number_buckets)]
                            for _ in range(self.number_hash_functions)]

    def lower(self, factor):
        for i in range(self.number_hash_functions):
            for j in range(self.number_buckets):
                self.count[i][j] *= factor


class RecordCategorialHash():
    def __init__(self, number_hash_functions, number_buckets, number_categorical_features):
        self.number_categorical_features = number_categorical_features
        self.number_buckets = number_buckets
        self.number_hash_functions = number_hash_functions
        self.clear()
        random.seed(datetime.now())

    def hash(self, x_categorical, i):
        resid = 0
        for k in range(self.number_categorical_features):
            resid = (resid + self.categorial_count[i][k] * x_categorical[k]) % self.number_buckets
        hashed_result = int(resid + (self.number_buckets if resid < 0 else 0))
        return hashed_result

    def clear(self):
        self.categorial_count = [[random.randrange(0, self.number_buckets - 1) + 1 if i < self.number_categorical_features - 1
                                  else random.randrange(0, self.number_buckets)
                                  for i in range(self.number_buckets)]
                                 for _ in range(self.number_hash_functions)]


class RecordNumericalHash():
    def __init__(self, number_hash_functions, number_buckets, number_numerical_features):
        self.number_numerical_features = number_numerical_features
        self.number_buckets = number_buckets
        self.number_hash_functions = number_hash_functions
        self.k = math.ceil(math.log2(self.number_buckets))
        self.rand_vectors = [np.random.normal(0, 1, size=(
            self.k, number_numerical_features)) for _ in range(number_hash_functions)]
        self.clear()

    def hash(self, x_numerical, i):
        x_numerical_vector = np.array(x_numerical)
        bitset = ""
        for j in range(self.k):
            dot_product = np.dot(x_numerical_vector, self.rand_vectors[i][j].T)
            if dot_product > 0:
                bitset += "1"
            else:
                bitset += "0"
        return int(bitset, 2)

    def clear(self):
        self.numerical_count = [[[0 for _ in range(self.number_buckets)] for _ in range(
            self.k)] for _ in range(self.number_hash_functions)]
