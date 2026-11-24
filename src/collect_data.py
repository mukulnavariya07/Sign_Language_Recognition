import argparse,csv
from collections import Counter
from pathlib import Path
import cv2
from .config import CSV,LABELS
from .features import normalize_landmarks
from .hand_detector import HandDetector

def main():
 p=argparse.ArgumentParser();p.add_argument('--frames',type=int,default=100);p.add_argument('--camera',type=int,default=0);p.add_argument('--output',type=Path,default=CSV);a=p.parse_args()
 a.output.parent.mkdir(parents=True,exist_ok=True);counts=Counter()
 if a.output.exists():
  with a.output.open(newline='') as f:counts.update(r['label'] for r in csv.DictReader(f))
 cap=cv2.VideoCapture(a.camera)
 if not cap.isOpened():raise RuntimeError('Webcam unavailable; check camera index and permissions')
 det=HandDetector();new=not a.output.exists() or a.output.stat().st_size==0
 try:
  while True:
   ok,frame=cap.read()
   if not ok:break
   frame=cv2.flip(frame,1);pts=det.detect(frame);det.draw(frame,pts)
   cv2.putText(frame,'Press A-Y (J excluded) to collect | Q quit',(8,25),0,.6,(255,255,255),2)
   cv2.putText(frame,'Counts '+str(dict(counts)),(8,55),0,.45,(0,255,255),1);cv2.imshow('Collect ASL',frame);k=cv2.waitKey(1)&255
   if k in (ord('q'),ord('Q')):break
   label=chr(k).upper() if k<128 else ''
   if label not in LABELS:continue
   rows=[]
   for i in range(a.frames):
    ok,fr=cap.read()
    if not ok:break
    fr=cv2.flip(fr,1);points=det.detect(fr);det.draw(fr,points)
    if points is not None:
     try:rows.append(normalize_landmarks(points))
     except ValueError:pass
    cv2.putText(fr,f'{label} {len(rows)}/{a.frames}',(10,30),0,.8,(0,255,0),2);cv2.imshow('Collect ASL',fr);cv2.waitKey(1)
   with a.output.open('a',newline='') as f:
    w=csv.writer(f)
    if new:w.writerow(['label']+[f'f{i}' for i in range(63)]);new=False
    for row in rows:w.writerow([label,*row])
   counts[label]+=len(rows);print(f'{label}: saved {len(rows)} valid frames')
 finally:cap.release();det.close();cv2.destroyAllWindows()
if __name__=='__main__':main()
