import numpy as np,pytest
from src.features import normalize_landmarks,mirror_features

def sample():
 p=np.random.default_rng(4).random((21,3),dtype=np.float32);p[0]=[.1,.1,.1];p[9]=[.7,.6,.5];return p
def test_translation_invariance():
 x=sample();np.testing.assert_allclose(normalize_landmarks(x),normalize_landmarks(x+[2,-4,8]),atol=1e-5)
def test_scale_invariance():
 x=sample();np.testing.assert_allclose(normalize_landmarks(x),normalize_landmarks(x*4),atol=1e-5)
def test_mirror_twice_identity():
 x=np.arange(63,dtype=np.float32);np.testing.assert_allclose(mirror_features(mirror_features(x)),x)
def test_wrong_shape():
 with pytest.raises(ValueError):normalize_landmarks(np.zeros((20,3)))
