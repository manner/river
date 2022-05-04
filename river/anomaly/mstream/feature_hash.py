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
                if(min == max):
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
            result += self.counts_to_anom(self.total_count[bucket], self.count[bucket], t)
        return result

    def hash(self, value):
        # this is currently more simplified than the cpp implementation
        return floor(value * self.number_buckets) % self.number_buckets

    def normalize(self, value, min, max):
        return (value - min) / (max - min)

    def clear(self):
        self.count = [[0 for i in range(self.number_buckets)] for j in range(self.number_features)]
        self.total_count = [[0 for i in range(self.number_buckets)]
                            for j in range(self.number_features)]

    def lower(self, factor):
        for i in range(self.number_features):
            for j in range(self.number_buckets):
                self.count[i][j] *= factor

    # This is directly copied from the cpp implementation
    # I don't know exactly where this function is coming from, as it's not in the paper
    def counts_to_anom(self, tot, cur, cur_t):
        cur_mean = tot / cur_t
        sqerr = pow(max(0, cur - cur_mean), 2)
        return sqerr / cur_mean + sqerr / (cur_mean * max(1, cur_t - 1))
