from datetime import datetime
from math import floor, log10
import random
import sys

# This is directly copied from the cpp implementation
# I don't know exactly where this function is coming from, as it's not in the paper
# TODO: This code shouldn't be duplicated in both feature and record hash


def counts_to_anom(tot, cur, cur_t):
    if tot == 0:
        return 0
    cur_mean = tot / cur_t
    sqerr = pow(max(0, cur - cur_mean), 2)
    a = sqerr / cur_mean
    b = sqerr / (cur_mean * max(1, cur_t - 1))
    return a + b


class FeatureHash():
    def __init__(self, number_buckets, number_hash_functions, feature_types, seed=datetime.now()):
        number_numerical_features = feature_types.count(False)
        number_categorical_features = feature_types.count(True)
        self.number_hash_functions = number_hash_functions
        self.numerical_hash = FeatureNumericalHash(
            number_buckets, number_numerical_features)
        self.categorial_hash = FeatureCategorialHash(
            number_buckets, self.number_hash_functions, number_categorical_features, seed)

    def insert(self, x_numerical, x_categorical):
        self.numerical_hash.insert(x_numerical)
        self.categorial_hash.insert(x_categorical)

    def get_count(self, x_numerical, x_categorical, timestamp):
        result = self.numerical_hash.get_count(x_numerical, timestamp)
        # print("Numerical FeatureHash", result)
        b = self.categorial_hash.get_count(x_categorical, timestamp)
        # print("Categorical FeatureHash", b)
        return result + b

    def lower(self, factor):
        self.numerical_hash.lower(factor)
        self.categorial_hash.lower(factor)


class FeatureCategorialHash():
    def __init__(self, number_buckets, number_hash_functions, number_features, s):
        # random.seed('2022-05-18 14:15:26.845156')

        seed = 1322998564246784757  # random.randrange(sys.maxsize)
        self.rng = random.Random(seed)

        print("Seed was: ", seed)
        self.number_buckets = number_buckets
        self.number_hash_functions = number_hash_functions
        self.number_features = number_features
        self.init_hash()
        self.clear()
        print(seed)

    def init_hash(self):
        self.hash1 = [[self.rng.randint(1, self.number_buckets - 1)
                      for _ in range(self.number_hash_functions)] for _ in range(self.number_features)]  # [1, p-1]
        self.hash2 = [[self.rng.randint(0, self.number_buckets - 1)
                      for _ in range(self.number_hash_functions)] for _ in range(self.number_features)]  # [0, p-1]

    def hash(self, feature, feature_index, j):
        resid = (int(feature) * self.hash1[feature_index][j] +
                 self.hash2[feature_index][j]) % self.number_buckets
        if resid < 0:
            return resid + self.number_buckets
        else:
            return resid

    def insert(self, x):
        for i, feature in enumerate(x):
            for j in range(self.number_hash_functions):
                bucket = self.hash(feature, i, j)
                self.count[i][j][bucket] += 1
                self.total_count[i][j][bucket] += 1

    def get_count(self, x, t):
        result = 0
        for i, feature in enumerate(x):
            min_count = float('inf')
            min_total_count = float('inf')
            for j in range(self.number_hash_functions):
                bucket = self.hash(feature, i, j)
                min_count = min(min_count, self.count[i][j][bucket])
                min_total_count = min(min_total_count, self.total_count[i][j][bucket])
            result += counts_to_anom(min_total_count, min_count, t)
        return result

    def clear(self):
        self.count = [[[0 for _ in range(self.number_buckets)] for _ in range(
            self.number_hash_functions)] for _ in range(self.number_features)]
        self.total_count = [[[0 for _ in range(self.number_buckets)] for _ in range(
            self.number_hash_functions)] for _ in range(self.number_features)]

    def lower(self, factor):
        for i in range(self.number_features):
            for j in range(self.number_hash_functions):
                for k in range(self.number_buckets):
                    self.count[i][j][k] *= factor


class FeatureNumericalHash():
    def __init__(self, number_buckets, number_features):
        self.number_buckets = number_buckets
        self.number_features = number_features
        self.number_observations = 0
        self.clear()

    def insert(self, x):
        for i, feature in enumerate(x):
            current_feature = log10(1 + feature)
            if self.number_observations == 0:
                self.min_features[i] = current_feature
                self.max_features[i] = current_feature
                current_feature = 0
            else:
                self.min_features[i] = min(self.min_features[i], current_feature)
                self.max_features[i] = max(self.max_features[i], current_feature)
                if self.min_features[i] == self.max_features[i]:
                    current_feature = 0
                else:
                    current_feature = self.normalize(
                        current_feature, self.min_features[i], self.max_features[i])

            bucket = self.hash(current_feature)
            self.count[i][bucket] += 1
            self.total_count[i][bucket] += 1
        self.number_observations += 1

    def get_count(self, x, t):
        result = 0
        for i, feature in enumerate(x):
            current_feature = log10(1 + feature)
            if self.number_observations == 0:
                current_feature = 0
            else:
                if self.min_features[i] == self.max_features[i]:
                    current_feature = 0
                elif current_feature > self.max_features[i]:
                    current_feature = 1
                elif current_feature < self.min_features[i]:
                    current_feature = 0
                else:
                    # print(
                    # f"c: {current_feature}, min: {self.min_features[i]}, max: {self.max_features[i]}")
                    current_feature = self.normalize(
                        current_feature, self.min_features[i], self.max_features[i])
            bucket = self.hash(current_feature)
            # print(f"tot: {self.total_count[i]}, bucket: {bucket}, current: {current_feature}")
            # print(self.total_count[i][bucket], self.count[i][bucket], t)
            result += counts_to_anom(self.total_count[i][bucket], self.count[i][bucket], t)
        return result

    def hash(self, value):
        bucket = floor(value * (self.number_buckets - 1))
        if bucket < 0:
            bucket = (bucket % self.number_buckets + self.number_buckets) % self.number_buckets
        return bucket

    def normalize(self, value, min, max):
        return (value - min) / (max - min)

    def clear(self):
        self.count = [[0 for _ in range(self.number_buckets)] for _ in range(self.number_features)]
        self.total_count = [[0 for _ in range(self.number_buckets)]
                            for _ in range(self.number_features)]
        self.min_features = [float('inf') for _ in range(self.number_features)]
        self.max_features = [float('-inf') for _ in range(self.number_features)]

    def lower(self, factor):
        for i in range(self.number_features):
            for j in range(self.number_buckets):
                self.count[i][j] *= factor
