"""MediaPipe Tasks detector. Import is intentionally deferred until instantiation."""
import urllib.request
from pathlib import Path
import cv2
from .config import TASK,TASK_URL
class HandDetector:
 def __init__(self,model_path=TASK,num_hands=1):
  try:
   import mediapipe as mp
   from mediapipe.tasks import python
   from mediapipe.tasks.python import vision
  except ImportError as e: raise RuntimeError('Install MediaPipe with pip install -r requirements.txt') from e
  path=Path(model_path)
  if not path.exists():
   path.parent.mkdir(parents=True,exist_ok=True)
   try: print('Downloading hand_landmarker.task ...'); urllib.request.urlretrieve(TASK_URL,path)
   except Exception as e:
    path.unlink(missing_ok=True); raise RuntimeError(f'Download failed. Download {TASK_URL} manually to {path}. Error: {e}') from e
  options=vision.HandLandmarkerOptions(base_options=python.BaseOptions(model_asset_path=str(path)),running_mode=vision.RunningMode.IMAGE,num_hands=num_hands)
  self.mp=mp; self.detector=vision.HandLandmarker.create_from_options(options)
 def detect(self,bgr):
  rgb=cv2.cvtColor(bgr,cv2.COLOR_BGR2RGB); image=self.mp.Image(image_format=self.mp.ImageFormat.SRGB,data=rgb)
  r=self.detector.detect(image)
  return None if not r.hand_landmarks else np_points(r.hand_landmarks[0])
 def draw(self,frame,points):
  if points is None:return frame
  h,w=frame.shape[:2]; q=[(int(x*w),int(y*h)) for x,y,z in points]
  edges=[(0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),(5,9),(9,10),(10,11),(11,12),(9,13),(13,14),(14,15),(15,16),(13,17),(17,18),(18,19),(19,20),(0,17)]
  for a,b in edges:cv2.line(frame,q[a],q[b],(70,220,70),2)
  for p in q:cv2.circle(frame,p,4,(0,180,255),-1)
  return frame
 def close(self):self.detector.close()
def np_points(points):return [[p.x,p.y,p.z] for p in points]
import numpy as np
