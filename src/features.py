import numpy as np

def normalize_landmarks(landmarks):
    """Wrist-center 21 xyz points and scale by wrist-to-MCP9 distance."""
    p=np.asarray(landmarks,dtype=np.float32)
    if p.shape!=(21,3): raise ValueError(f'Expected (21,3), got {p.shape}')
    if not np.isfinite(p).all(): raise ValueError('Landmarks must be finite')
    p=p-p[0]; scale=float(np.linalg.norm(p[9]))
    if scale<1e-8: raise ValueError('Degenerate wrist-to-landmark-9 scale')
    return (p/scale).reshape(63).astype(np.float32)

def mirror_features(features):
    x=np.asarray(features,dtype=np.float32)
    if x.shape[-1]!=63: raise ValueError('Last dimension must be 63')
    y=x.reshape(*x.shape[:-1],21,3).copy(); y[...,0]*=-1
    return y.reshape(x.shape)
