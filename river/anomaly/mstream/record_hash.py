import math
import random
from itertools import compress
import numpy as np


class RecordHash():
    def __init__(self, is_categorical, number_buckets):
        self.is_categorical = is_categorical
        self.is_numerical = [not feature for feature in is_categorical]
        # self.num_rows = num_rows number of hash functions, we will just use 2 for now
        self.number_buckets = number_buckets
        self.number_numerical_features = len(self.is_numerical)
        self.number_categorical_features = len(self.is_categorical)
        self.numerical_hash = RecordNumericalHash(
            self.number_buckets, self.number_numerical_features)
        self.categorical_hash = RecordCategorialHash(
            self.number_buckets, self.number_categorical_features)

    def insert(self, x):
        # divide x into categorical and numerical
        x_categorical = compress(x, self.is_categorical)
        x_numerical = compress(x, self.is_numerical)
        self.categorical_hash.hash(x_categorical)
        self.numerical_hash.hash(x_numerical)

    def get_count(self):
        return (self.categorical_hash.get_count(), self.numerical_hash.get_count()) % self.number_buckets


class RecordCategorialHash():
    def __init__(self, num_buckets, number_categorical_features):
        self.cat_recordhash = []
        self.number_categorical_features = number_categorical_features
        self.num_buckets = num_buckets
        for i in range(self.number_categorical_features):
            self.cat_recordhash.append({})
        self.clear()

    def hash(self, x_categorical):
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

    def get_count(self):
        return self.bucket_cat


class RecordNumericalHash():
    def __init__(self, num_buckets, number_numerical_features):
        self.number_numerical_features = number_numerical_features
        self.num_buckets = num_buckets
        self.k = math.ceil(math.log2(self.num_buckets))
        p = number_numerical_features
        self.rand_vectors = np.random.normal(0, 1, size=(self.k, p))
        self.clear()

    def hash(self, x_numerical):
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

    def get_count(self):
        return self.bucket_num
