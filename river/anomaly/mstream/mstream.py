from math import log

from ..base import AnomalyDetector
from .feature_hash import FeatureHash
from .record_hash import RecordHash

from itertools import compress

__all__ = ["MStream"]


class MStream(AnomalyDetector):
    """
    MStream implementation

    Description:

    Parameters:

    Examples:

    """

    def __init__(self, feature_types, number_buckets=1024, factor=0.8, number_hash_functions=2, timestamp_key='timestamp'):
        # is_categorical is just a vector that says True if a feature is categorical, and false if it's numerical
        # this assumes an entry will always have the same order in the features
        # ex. [True, False, False, True, False]
        self.categorial_features = feature_types
        self.numerical_features = [not x for x in feature_types]
        self.record_hash = RecordHash(feature_types, number_buckets, number_hash_functions)
        # todo ad feature_types parameter
        self.feature_hash = FeatureHash(number_buckets, number_hash_functions)
        self.factor = factor
        self.is_empty = True
        self.timestamp_key = timestamp_key

    def learn_one(self, x):
        (timestamp, x_numerical, x_categorical) = self.get_values(x)
        if self.is_empty or timestamp > self.current_time:
            self.record_hash.lower(self.factor)
            self.feature_hash.lower(self.factor)
            self.current_time = timestamp
            self.is_empty = False
        self.record_hash.insert(x_categorical, x_numerical)
        self.feature_hash.insert(x_categorical, x_numerical)
        return self

    def score_one(self, x):
        (timestamp, x_numerical, x_categorical) = self.get_values(x)
        score = self.record_hash.get_count(x_numerical, x_categorical, timestamp)
        score += self.feature_hash.get_count(x_numerical, x_categorical, timestamp)
        return log(1 + score)

    def get_values(self, x):
        timestamp = self.get_timestamp(x)
        values = list(x.values())
        values = values[:-1]
        x_numerical = compress(values, self.numerical_features)
        x_categorical = compress(values, self.categorial_features)
        return (timestamp, x_categorical, x_numerical)

    def get_timestamp(self, x):
        return x[self.timestamp_key]
