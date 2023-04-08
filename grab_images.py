#=============================================================================================================#
#
#=============================================================================================================#
import flickrapi
import urllib
import cv2
import dlib
from imutils import face_utils
import pandas as pd
import numpy as np

import os, sys, urllib
import multiprocessing as mp

os.makedirs('C:/Jupyter/edges2portrait_gan-master/edges2portrait', exist_ok=True)
os.makedirs('C:/Jupyter/edges2portrait_gan-master/edges2portrait/trainA', exist_ok=True)
os.makedirs('C:/Jupyter/edges2portrait_gan-master/edges2portrait/trainB', exist_ok=True)

#=============================================================================================================#
#	
#=============================================================================================================#
facenet = cv2.dnn.readNetFromTensorflow(
  'C:/Jupyter/edges2portrait_gan-master/face_models/opencv_face_detector_uint8.pb',
  'C:/Jupyter/edges2portrait_gan-master/face_models/opencv_face_detector.pbtxt'
)

net = cv2.dnn.readNetFromTensorflow('C:/Jupyter/edges2portrait_gan-master/face_models/landmarks_net.pb')

#predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

FACE_IMG_SIZE = 256
FACE_THRESHOLD = 0.7
KEYWORD = 'portrait  bldigital'

#=============================================================================================================#
#
#=============================================================================================================#
flickr = flickrapi.FlickrAPI('c6a2c45591d4973ff525042472446ca2', '202ffe6f387ce29b', cache=True)

photos = flickr.walk(
  text=KEYWORD,
  tag_mode='all',
  # tags=KEYWORD,
  extras='url_c',
  per_page=100,
  sort='relevance',
)

#=============================================================================================================#
#
#=============================================================================================================#
#def url_to_img(url):
#   resp = urllib.request.urlopen(url)
#   img = np.asarray(bytearray(resp.read()), dtype=np.uint8)
#   img = cv2.imdecode(img, cv2.IMREAD_COLOR)
#   return img

#=============================================================================================================#
#
#=============================================================================================================#
def imgfile_to_img(img_pathname):
    with open(img_pathname, 'rb') as f:
      data = f.read()
    img = np.asarray(bytearray(data), dtype=np.uint8)
    img = cv2.imdecode(img, cv2.IMREAD_COLOR)
    return img
#=============================================================================================================#
#
#=============================================================================================================#
def process(img_path):
  try:
    #img_pname = img_path + '.png'           ####
    img_pname = img_path                  ####
    img = imgfile_to_img(img_pname)

    filename = os.path.basename(img_pname)

    blob = cv2.dnn.blobFromImage(img, scalefactor=1.0, size=(300, 300), mean=[104, 117, 123], swapRB=False, crop=False)

    facenet.setInput(blob)
    dets = facenet.forward()

    face_img = None

    for i in range(dets.shape[2]):
      conf = dets[0, 0, i, 2]

      if conf < FACE_THRESHOLD:
        continue

      x1 = dets[0, 0, i, 3] * img.shape[1]
      y1 = dets[0, 0, i, 4] * img.shape[0]
      x2 = dets[0, 0, i, 5] * img.shape[1]
      y2 = dets[0, 0, i, 6] * img.shape[0]
      
      face_length = max(x2 - x1, y2 - y1)
      cx, cy = (x1 + x2) / 2, (y1 + y2) / 2   # 이미지 파일의 중점

      sx1 = int(cx - face_length * 0.2)
      sx2 = int(cx + face_length * 0.2)
      sy1 = int(cy - face_length * 0.2)
      sy2 = int(cy + face_length * 0.2)
    
      if sx1 <= 0 or sy1 <= 0:
        return False

      #if x1 <= 0 or y1 <= 0:
      #  return False

      face_img = img[sy1:sy2, sx1:sx2]
      #face_img = img[y1:y2, x1:x2]
      face_img = cv2.resize(face_img, (FACE_IMG_SIZE, FACE_IMG_SIZE), interpolation=cv2.INTER_LINEAR)
      face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)

      margin = int(FACE_IMG_SIZE * 0.25)

      color = (0)
      cv2.rectangle(face_img, pt1=(margin, margin), pt2=(FACE_IMG_SIZE-margin, FACE_IMG_SIZE-margin), thickness=2, color=color)

      # landmarks
      #shape = predictor(face_img, dlib.rectangle(left=margin, top=margin, right=FACE_IMG_SIZE-margin, bottom=FACE_IMG_SIZE-margin))
      #contours = face_utils.shape_to_np(shape)

      """
      # draw polylines
      #empty_canvas = np.zeros((FACE_IMG_SIZE, FACE_IMG_SIZE, 3), dtype=np.uint8)


      lines = {
        'leb_line': [contours[17:(21+1)], (255, 255, 255), False], # left eyebrow
        'reb_line': [contours[22:(26+1)], (255, 255, 255), False], # right eyebrow
        'le_line': [contours[42:(47+1)], (255, 255, 255), True], # le
        're_line': [contours[36:(41+1)], (255, 255, 255), True], # re
        'nose_line_1': [contours[27:(30+1)], (255, 255, 255), False], # nose vertical
        'nose_line_2': [contours[31:(35+1)], (255, 255, 255), False], # nose horizontal
        'jaw_line': [contours[0:(16+1)], (255, 255, 255), False], # jaw
        'lip_line': [contours[48:(67+1)], (255, 255, 255), True] # lip
      }

      C = cv2.cvtColor(face_img.copy(), cv2.COLOR_GRAY2BGR)

      cv2.rectangle(C, pt1=(margin, margin), pt2=(FACE_IMG_SIZE-margin, FACE_IMG_SIZE-margin), thickness=2, color=255)

      for key, line in lines.items():
        cv2.polylines(empty_canvas, [line[0].astype(int)], isClosed=line[2], color=line[1], thickness=2)
        # cv2.polylines(C, [line[0].astype(int)], isClosed=line[2], color=(255,0,0), thickness=2)
      """

      cv2.imshow('result', face_img)

      #cv2.imwrite(os.path.join('C:/Jupyter/edges2portrait_gan-master/edges2portrait/trainA', filename), empty_canvas)
      cv2.imwrite(os.path.join('C:/Jupyter/edges2portrait_gan-master/edges2portrait/trainB', filename), face_img)
      # cv2.imwrite(os.path.join('edges2portrait/trainB', filename), C)
      key = cv2.waitKey(10000)

      break

    if face_img is None:
      return False

  except:
    print('[!] Error saving image', img_path)

  return {'file': filename, 'url': img_path}

n = 0

#=============================================================================================================#
#
#=============================================================================================================#
def cb(x):
  global n, df

  if x is not False:
    df = df.append(x, ignore_index=True)

  n += 1
  sys.stdout.write('\r%s/%s' % (n, '?'))
  sys.stdout.flush()

#=============================================================================================================#
#
#=============================================================================================================#
if __name__ == '__main__':
  df = pd.DataFrame(columns=['file', 'url'])

  pool = mp.Pool(processes=mp.cpu_count())

  results = []

  data_path = 'C:/Jupyter/edges2portrait_gan-master/facephoto/WOM/' ####

  cnt = 0

  ###for i, photo in enumerate(photos):
  for i in os.listdir(data_path):
    img_path = data_path + i            ####
    # print(img_path)                   ####

    if cnt > 10:
      break

    ###url = photo.get('url_c')

    if (i == 0) or (img_path is None):
      continue

    result = pool.apply_async(process, (img_path,), callback=cb)

    results.append(result)

    cnt += 1

  for r in results:
    r.wait()

  df.to_csv(os.path.join('edges2portrait', 'metadata.csv'), index=False, header=False)

  print('saved %d images!' % len(df))
