##===================================================================================##
## 필요한 모듈 입력
##===================================================================================##
from skimage.io import imread
from skimage.transform import resize
from glob import glob
import numpy as np
import matplotlib.pyplot as plt
import os

##===================================================================================##
## DataGenerator() 클래스 정의
##===================================================================================##
class DataGenerator():
  def __init__(self):
    pass

  ##=================================================================================##
  # process() 함수
  ##=================================================================================##
  def process(self, batch_path, is_train):
    imgs_A, imgs_B = [], []

    ##-------------------------------------------------------------------------------##
    # 배치경로에서 이미지를 입력하여 어레이 생성
    ##-------------------------------------------------------------------------------##
    for img_path in batch_path:
      img_A = imread(img_path, as_gray=True)               # 랜드마크 이미지 입력
      img_B = imread(os.path.join('edges2portrait/trainB', os.path.basename(img_path)), as_gray=True) # 랜드마크의 소스이미지 입

      if is_train and np.random.random() < 0.5:
        img_A = np.fliplr(img_A)
        img_B = np.fliplr(img_B)

      imgs_A.append(np.expand_dims(img_A, axis=-1))
      imgs_B.append(np.expand_dims(img_B, axis=-1))

    imgs_A = np.array(imgs_A) / 127.5 - 1.
    imgs_B = np.array(imgs_B) / 127.5 - 1.

    return imgs_A, imgs_B

  ##=================================================================================##
  # load_data() 함수
  ##=================================================================================##
  def load_data(self, batch_size=1, is_train=True):
    listA = glob('edges2portrait/trainA/*.jpg')             # 랜드마크 이미지의 파일명을 리스트로 반환

    batch_path = np.random.choice(listA, size=batch_size)

    imgs_A, imgs_B = self.process(batch_path, is_train)

    return imgs_A, imgs_B

  ##=================================================================================##
  # load_batch() 함수
  ##=================================================================================##
  def load_batch(self, batch_size=1, is_train=True):
    listA = glob('edges2portrait/trainA/*.jpg')             # 랜드마크 이미지의 파일명을 리스트로 반환

    self.n_batches = int(len(listA) / batch_size)

    for i in range(self.n_batches-1):
      batch_path = listA[i*batch_size:(i+1)*batch_size]
      
      imgs_A, imgs_B = self.process(batch_path, is_train)

      yield imgs_A, imgs_B

if __name__ == '__main__':
  dg = DataGenerator()
  a = dg.load_data(batch_size=3, is_train=True)

  print(a)
