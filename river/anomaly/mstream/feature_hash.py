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
    def __init__(self):
        pass

    def insert(self, x):
        pass
