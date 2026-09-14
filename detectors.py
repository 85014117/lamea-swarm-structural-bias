import hashlib
import sys
import settings as cfg
import numpy as np

def seedTool(aid):
    from rpy2 import robjects
    seed = int.from_bytes(hashlib.sha256(aid.encode()).digest()[:4], "little")
    np.random.seed(seed)
    robjects.r["set.seed"](seed % 2147483647) # clip int limit

def makeBias():
    sys.path.insert(0, str(cfg.biasDir))
    import tensorflow as tf
    from BIAS import BIAS
    tf.config.set_visible_devices([], "GPU")
    return BIAS()
