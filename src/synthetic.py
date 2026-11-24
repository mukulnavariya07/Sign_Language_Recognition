"""Generate unmistakably FAKE vectors for pipeline smoke tests only."""
import argparse
from pathlib import Path
import numpy as np,pandas as pd
from .config import CSV,LABELS

def make_fake_csv(path=CSV,samples_per_class=24,seed=23):
 rng=np.random.default_rng(seed);prototypes=rng.normal(0,.65,(len(LABELS),63));rows=[]
 for i,label in enumerate(LABELS):
  for _ in range(samples_per_class):rows.append([label,*((prototypes[i]+rng.normal(0,.1,63)).tolist())])
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True);pd.DataFrame(rows,columns=['label']+[f'f{i}' for i in range(63)]).to_csv(path,index=False)
 print(f'FAKE synthetic dataset saved: {len(rows)} rows at {path}; NOT real ASL samples.')
def main():
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=CSV);p.add_argument('--samples',type=int,default=24);p.add_argument('--seed',type=int,default=23);a=p.parse_args();make_fake_csv(a.output,a.samples,a.seed)
if __name__=='__main__':main()
