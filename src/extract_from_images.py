import argparse,csv
from pathlib import Path
import cv2
from .config import CSV,LABELS
from .features import normalize_landmarks
from .hand_detector import HandDetector

def main():
 p=argparse.ArgumentParser(description='Extract from root/LETTER/image files');p.add_argument('root',type=Path);p.add_argument('--output',type=Path,default=CSV);a=p.parse_args()
 a.output.parent.mkdir(parents=True,exist_ok=True);det=HandDetector();header=not a.output.exists() or a.output.stat().st_size==0
 try:
  with a.output.open('a',newline='') as f:
   w=csv.writer(f)
   if header:w.writerow(['label']+[f'f{i}' for i in range(63)])
   total=kept=0
   for label in LABELS:
    paths=[p for p in (a.root/label).glob('*') if p.suffix.lower() in ('.jpg','.jpeg','.png','.bmp')];n=m=0
    for path in paths:
     n+=1;total+=1;image=cv2.imread(str(path))
     if image is None:continue
     points=det.detect(image)
     if points is None:continue
     try:features=normalize_landmarks(points)
     except ValueError:continue
     w.writerow([label,*features]);m+=1;kept+=1
    print(f'{label}: kept {m}/{n}')
  print(f'Overall kept {kept}/{total}; output {a.output}')
 finally:det.close()
if __name__=='__main__':main()
