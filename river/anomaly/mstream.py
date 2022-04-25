from .base import AnomalyDetector

import random


class MStream(AnomalyDetector):
    """ 
    MStream implementation

    Description:

    Parameters:

    Examples:

    """

    def __init__(self, seed: int = None):
        self.record_hash = RecordHash()
        self.feature_hash = FeatureHash(5, seed)

    def learn_one(self, x):
        # TODO
        self.record_hash.insert(x)
        self.feature_hash.insert(x)
        return self

    def score_one(self, x):
        score = self.record_hash.score(x)
        score += self.feature_hash.score(x)
        return score


class RecordHash():
    def __init__(self):
        pass

    def insert(self, x):
        pass


class FeatureHash():
    def __init__(self, number_buckets, seed):
        self.numerical_hash = NumericalHash()
        self.categorial_hash = CategorialHash(number_buckets, seed)
        pass

    def insert(self, x):
        # categorial hash
        self.categorial_hash.insert(x)
        # real-value hash
        self.numerical_hash.insert(x)
        pass


class CategorialHash():
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


class NumericalHash():
    def __init__(self):
        pass

    def insert(self, x):
        pass
