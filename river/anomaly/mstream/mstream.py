from math import log

from ..base import AnomalyDetector
from .feature_hash import FeatureHash
from .record_hash import RecordHash

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
        self.feature_types = feature_types
        self.record_hash = RecordHash(feature_types, number_buckets, number_hash_functions)
        # todo ad feature_types parameter
        self.feature_hash = FeatureHash(number_buckets, number_hash_functions)
        self.factor = factor
        self.is_empty = True

    def learn_one(self, x):
        ts = self.get_timestamp(x)  # not sure how to get timestamp from observation
        if self.is_empty or ts > self.current_time:
            self.record_hash.lower(self.factor)
            self.feature_hash.lower(self.factor)
            self.current_time = ts
            self.is_empty = False
        self.record_hash.insert(x)
        self.feature_hash.insert(x)
        return self

    def score_one(self, x):
        ts = self.get_timestamp(x)  # not sure how to get timestamp from observation
        score = self.record_hash.get_count(x, ts)  # no params for now, working on this
        score += self.feature_hash.get_count(x, ts)
        return log(1 + score)

    def get_timestamp(self, x):
        return x[self.timestamp_key]
