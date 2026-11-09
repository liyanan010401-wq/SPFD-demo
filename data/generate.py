"""SYNTHETIC ONLY: artificial rectangles and sinusoidal textures, never real sonar."""
from pathlib import Path
import os
import numpy as np

def generate(path,seed=1729):
    rng=np.random.default_rng(seed); n=24; h=w=24
    images=np.zeros((n,1,h,w),np.float32);boxes=np.zeros((n,2,4),np.float32);counts=np.arange(n)%3
    valid=np.ones((n,1,h,w),bool);valid[:,:,-2:]=False
    yy,xx=np.mgrid[:h,:w]
    for i in range(n):
        x=.13+.045*rng.normal(size=(h,w))+.025*np.sin(xx*(.45 if i<18 else .8))
        for j in range(counts[i]):
            cx=int(rng.integers(5,19));cy=5+j*10; bw=8;bh=3
            x[(abs(xx-cx)<=bw/2)&(abs(yy-cy)<=bh/2)]+=.5
            boxes[i,j]=[cx/w,cy/h,bw/w,bh/h]
        images[i,0]=np.clip(x,0,1);images[i,0,-2:]=0
    split=np.array(['train']*12+['dev']*6+['test']*6)
    np.savez(path,images=images,boxes=boxes,counts=counts,valid=valid,split=split,group_id=np.arange(n)//2,window_id=np.arange(n),frequency=np.where(np.arange(n)<18,1,2), synthetic_data=True, publication_evidence=False, dataset_id='SYNTHETIC-SPFD-DEMO-v2')
if __name__=='__main__':
    p=Path(__file__).absolute().with_name('SYNTHETIC_sample.npz')
    if os.name=='nt' and not str(p).startswith('\\\\?\\'):
        p=Path('\\\\?\\'+str(p))
    generate(p)
    print('Generated SYNTHETIC data only: '+str(p))
