from datetime import datetime
from math import floor, log10
import random

# This is directly copied from the cpp implementation
# I don't know exactly where this function is coming from, as it's not in the paper
# TODO: This code shouldn't be duplicated in both feature and record hash


def counts_to_anom(tot, cur, cur_t):
    cur_mean = tot / cur_t
    sqerr = pow(max(0, cur - cur_mean), 2)
    return sqerr / cur_mean + sqerr / (cur_mean * max(1, cur_t - 1))


class FeatureHash():
    def __init__(self, number_buckets, seed=datetime.now()):
        number_numerical_features = 5  # tbd
        number_categorical_features = 5  # tbd
        self.numerical_hash = FeatureNumericalHash(number_buckets, number_numerical_features)
        self.categorial_hash = FeatureCategorialHash(
            number_buckets, number_categorical_features, seed)

    def insert(self, x):
        # categorial hash
        self.categorial_hash.insert(x)
        # real-value hash
        self.numerical_hash.insert(x)

    def get_count(self, x, t):
        result = self.numerical_hash.get_count(x, t)
        result += self.categorial_hash.get_count(x, t)
        return result

    def lower(self, factor):
        self.categorial_hash.lower(factor)
        self.numerical_hash.lower(factor)


class FeatureCategorialHash():
    def __init__(self, number_buckets, number_features, seed):
        random.seed(seed)
        self.number_buckets = number_buckets
        self.number_features = number_features
        self.init_hash()
        self.clear()

    def init_hash(self):
        self.hash1 = [random.randrange(1, self.number_buckets - 1)
                      for _ in range(self.number_features)]  # [1, p-1]
        self.hash2 = [random.randrange(0, self.number_buckets - 1)
                      for _ in range(self.number_features)]  # [0, p-1]

    def hash(self, feature, i):
        resid = (feature * self.hash1[i] + self.hash2[i]) % self.number_buckets
        if resid < 0:
            return resid + self.number_buckets
        else:
            return resid

    def insert(self, x):
        for i, feature in enumerate(x):
            for j in range(self.number_features):
                bucket = self.hash(feature, j)
                self.count[i][j][bucket] += 1
                self.total_count[i][j][bucket] += 1

    def get_count(self, x, t):
        result = 0
        for i, feature in enumerate(x):
            min_count = float('inf')
            min_total_count = float('inf')
            for j in range(self.number_features):
                bucket = self.hash(feature, j)
                min_count = min(min_count, self.count[i][j][bucket])
                min_total_count = min(min_total_count, self.total_count[i][j][bucket])
            result += counts_to_anom(min_total_count, min_count, t)
        return result

    def clear(self):
        self.count = [[[0 for _ in range(self.number_buckets)] for _ in range(
            self.number_features)] for _ in range(self.number_features)]
        self.total_count = [[[0 for _ in range(self.number_buckets)] for _ in range(
            self.number_features)] for _ in range(self.number_features)]

    def lower(self, factor):
        for i in range(self.number_features):
            for j in range(self.number_features):
                for k in range(self.number_buckets):
                    self.count[i][j][k] *= factor


class FeatureNumericalHash():
    def __init__(self, number_buckets, number_features):
        self.number_buckets = number_buckets
        self.number_features = number_features
        self.clear()

    def insert(self, x):
        first = True
        for i, feature in enumerate(x):
            current_feature = log10(1 + feature)
            if first:
                first = False
                min = current_feature
                max = current_feature
                current_feature = 0
            else:
                min = min(min, current_feature)
                max = max(max, current_feature)
                if min == max:
                    current_feature = 0
                else:
                    current_feature = self.normalize(current_feature, min, max)

            bucket = hash(current_feature)
            self.count[i][bucket] += 1
            self.total_count[i][bucket] += 1

    def get_count(self, x, t):
        result = 0
        for feature in x:
            bucket = self.hash(feature)
            result += counts_to_anom(self.total_count[bucket], self.count[bucket], t)
        return result

    def hash(self, value):
        # this is currently more simplified than the cpp implementation
        return floor(value * self.number_buckets) % self.number_buckets

    def normalize(self, value, min, max):
        return (value - min) / (max - min)

    def clear(self):
        self.count = [[0 for _ in range(self.number_buckets)] for _ in range(self.number_features)]
        self.total_count = [[0 for _ in range(self.number_buckets)]
                            for _ in range(self.number_features)]

    def lower(self, factor):
        for i in range(self.number_features):
            for j in range(self.number_buckets):
                self.count[i][j] *= factor
