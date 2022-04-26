from ..base import AnomalyDetector
from .feature_hash import FeatureHash
from .record_hash import RecordHash


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
