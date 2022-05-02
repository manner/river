from math import floor, log10
import random


class FeatureHash():
    def __init__(self, number_buckets, seed):
        self.numerical_hash = FeatureNumericalHash()
        self.categorial_hash = FeatureCategorialHash(number_buckets, seed)
        pass

    def insert(self, x):
        # categorial hash
        self.categorial_hash.insert(x)
        # real-value hash
        self.numerical_hash.insert(x)
        pass


class FeatureCategorialHash():
    def __init__(self, number_buckets, seed):
        random.seed(seed)
        self.number_buckets = number_buckets
        self.hash1 = []
        self.hash2 = []

    def add_new_hash(self):
        self.hash1.append(random.randrange(1, self.number_buckets - 1))
        self.hash2.append(random.randrange(0, self.number_buckets))

    def hash(self, x, i):
        resid = (x * self.hash1[i] + self.hash2[i]) % self.num_buckets

    def insert(self, x):
        pass


class FeatureNumericalHash():
    def __init__(self, number_buckets, number_features):
        self.number_buckets = number_buckets
        self.number_features = number_features
        self.count = [[0 for i in range(number_buckets)] for j in range(number_features)]
        self.total_count = [[0 for i in range(number_buckets)] for j in range(number_features)]

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
                if(min == max):
                    current_feature = 0
                else:
                    current_feature = self.normalize(current_feature, min, max)

            bucket = hash(current_feature)
            self.count[i][bucket] += 1
            self.total_count[i][bucket] += 1

    def get_count(self, x):
        bucket = self.hash(x)
        return self.count[bucket]

    def hash(self, value):
        return floor(value * self.number_buckets) % self.number_buckets

    def normalize(self, value, min, max):
        return (value - min) / (max - min)
