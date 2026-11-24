"""CPU webcam inference with confidence gate, voting, and hold-to-type."""
import argparse,time,joblib,cv2
from collections import deque,Counter
from pathlib import Path
import numpy as np
from .config import MODEL
from .features import normalize_landmarks
from .hand_detector import HandDetector

class HoldSpeller:
 def __init__(self,hold_seconds=.5):self.hold_seconds=hold_seconds;self.candidate=None;self.started=None;self.committed=None;self.text=''
 def update(self,label,now=None):
  now=time.monotonic() if now is None else now
  if label is None:self.candidate=self.started=self.committed=None;return None
  if label!=self.candidate:self.candidate=label;self.started=now;self.committed=None
  if self.committed is None and now-self.started>=self.hold_seconds:self.text+=label;self.committed=label;return label
  return None
 def clear(self):self.text=''

def main():
 p=argparse.ArgumentParser();p.add_argument('--model',type=Path,default=MODEL);p.add_argument('--camera',type=int,default=0);p.add_argument('--threshold',type=float,default=.65);p.add_argument('--hold',type=float,default=.5);p.add_argument('--window',type=int,default=7);a=p.parse_args()
 if not a.model.exists():raise FileNotFoundError(f'Model absent: {a.model}. Collect data, then run python -m src.train')
 model=joblib.load(a.model)['model'];det=HandDetector();cap=cv2.VideoCapture(a.camera)
 if not cap.isOpened():det.close();raise RuntimeError('Cannot open webcam; verify camera index/permissions')
 votes=deque(maxlen=max(1,a.window));speller=HoldSpeller(a.hold)
 try:
  while True:
   ok,frame=cap.read()
   if not ok:break
   frame=cv2.flip(frame,1);points=det.detect(frame);det.draw(frame,points);label=None;conf=0.
   if points is not None:
    try:
     prob=model.predict_proba(normalize_landmarks(points).reshape(1,-1))[0];i=int(np.argmax(prob));conf=float(prob[i]);guess=str(model.classes_[i])
     if conf>=a.threshold:votes.append(guess)
     else:votes.clear()
     if votes:
      vote,n=Counter(votes).most_common(1)[0]
      if n>=max(1,(len(votes)+1)//2):label=vote
    except ValueError:votes.clear()
   else:votes.clear()
   typed=speller.update(label)
   cv2.putText(frame,f'Letter: {label or "-"}  confidence {conf:.2f}',(10,30),0,.7,(0,255,0),2)
   cv2.putText(frame,'Text: '+speller.text,(10,70),0,.85,(255,255,255),2)
   cv2.putText(frame,'Hold 0.5s | SPACE backspace | C clear | Q quit',(10,110),0,.5,(0,255,255),1)
   if typed:cv2.putText(frame,'COMMITTED '+typed,(10,150),0,.8,(0,255,255),2)
   cv2.imshow('ASL Alphabet',frame);key=cv2.waitKey(1)&255
   if key in (ord('q'),ord('Q')):break
   if key==ord('c'):speller.clear()
   if key==32:speller.text=speller.text[:-1]
 finally:cap.release();det.close();cv2.destroyAllWindows()
if __name__=='__main__':main()
