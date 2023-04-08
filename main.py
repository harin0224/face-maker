##===================================================================================##
## 필요한 모듈 입력
##===================================================================================##
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, Concatenate
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
from keras.layers import LeakyReLU
from keras.layers import UpSampling2D, Conv2D
from keras.models import Model
from keras.optimizers import Adam
import matplotlib.pyplot as plt
import numpy as np

import datetime, sys, os

##===================================================================================##
## DataGenerator 사용자 정의 모듈 입력
##  - 실제 데이터를 생성하는데 필요한 클래스를 정의함.
##===================================================================================##
from DataGenerator import DataGenerator

class Pix2Pix():
  ##=================================================================================##
  # 초기화 함수
  ##=================================================================================##
  def __init__(self):
    self.img_rows = 256               # 디폴트 이미지 세로크기 정의
    self.img_cols = 256               # 디폴트 이미지 가로크기 정의
    self.channels = 1                 # 디폴트 이미지 색상타입 정의 (1 : 흑백)
    self.img_shape = (self.img_rows, self.img_cols, self.channels)

    self.data_loader = DataGenerator() # DataGenerator() 에 대한 생성자

    ##-------------------------------------------------------------------------------##
    # 판별자(PatchGAN) 의 출력 형태 계산
    ##-------------------------------------------------------------------------------##
    patch = int(self.img_rows / 2**4)
    self.disc_patch = (patch, patch, 1)

    ##-------------------------------------------------------------------------------##
    # 생성자와 판별자의 첫번째 레이어에서 사용되는 필터의 모양(숫자)정의
    ##-------------------------------------------------------------------------------##
    self.gf = 64                     # 생성자 필터
    self.df = 64                     # 판별자 필터

    ##-------------------------------------------------------------------------------##
    #  판별자 생성 
    #  - loss 함수 : mse
    #  - optimizer 타입 : adam
    ##-------------------------------------------------------------------------------##
    self.discriminator = self.build_discriminator()
    self.discriminator.compile(
      loss='mse',
      optimizer='adam',
      metrics=['accuracy']
    )

    ##-------------------------------------------------------------------------------##
    #  생성자 생성
    ##-------------------------------------------------------------------------------##
    self.generator = self.build_generator()

    img_A = Input(shape=self.img_shape)
    img_B = Input(shape=self.img_shape)

    fake_B = self.generator(img_A)        # img_A 폴더의 이미지를 기반으로 fake 이미지 생성

    self.discriminator.trainable = False  # 생성자를 훈련시키는 동안 판별자는 훈련하지 않음.

    valid = self.discriminator([img_A, fake_B]) # 생성된 fake 이미지와 img_A의 이미지를 비교 학습.


    ##-------------------------------------------------------------------------------##
    #  Model Combine
    #  - loss 함수 : mse, mae
    #  - loss 가중치 : 1 ~ 100
    #  - optimizer 타입 : adam
    ##-------------------------------------------------------------------------------##
    self.combined = Model(inputs=[img_A, img_B], outputs=[valid, fake_B])
    self.combined.compile(
      loss=['mse', 'mae'],
      loss_weights=[1, 100],
      optimizer='adam'
    )

  ##=================================================================================##
  #  생성자 함수 정의
  #  - 생성자 함수는 두가지 타입의 2차원 convolution layer를 사용하여 구성
  ##=================================================================================##
  def build_generator(self):

    ##-------------------------------------------------------------------------------##
    #  첫번째 2차원 convolution layer : 다운샘플링에 사용
    #  - 사용함수 :  LeakyReLU
    ##-------------------------------------------------------------------------------##
    def conv2d(layer_input, filters, kernel_size=4, bn=True):
      d = Conv2D(filters, kernel_size=kernel_size, strides=2, padding='same')(layer_input)
      d = LeakyReLU(alpha=0.2)(d)
      if bn:
        d = BatchNormalization(momentum=0.8)(d)
      return d

    ##-------------------------------------------------------------------------------##
    #  두번째 2차원 convolution layer : 업샘플링에 사용
    #  - 사용함수 : Dropout
    #  - 출력결과는 Concatenate 하여 사
    ##-------------------------------------------------------------------------------##
    def deconv2d(layer_input, skip_input, filters, kernel_size=4, dropout_rate=0):
      u = UpSampling2D(size=2)(layer_input)
      u = Conv2D(filters, kernel_size=kernel_size, strides=1, padding='same', activation='relu')(u)
      if dropout_rate:
        u = Dropout(dropout_rate)(u)
      u = BatchNormalization(momentum=0.8)(u)
      u = Concatenate()([u, skip_input])
      return u

    # 이미지 입력
    d0 = Input(shape=self.img_shape)

    ##-------------------------------------------------------------------------------##
    #  다운샘플링 과정에서 사용되는 convolution layer의 적층
    ##-------------------------------------------------------------------------------##
    d1 = conv2d(d0, self.gf, bn=False) # 첫번째 타입의 convolution layer를 사용
    d2 = conv2d(d1, self.gf*2)
    d3 = conv2d(d2, self.gf*4)
    d4 = conv2d(d3, self.gf*8)
    d5 = conv2d(d4, self.gf*8)
    d6 = conv2d(d5, self.gf*8)
    d7 = conv2d(d6, self.gf*8)

    ##-------------------------------------------------------------------------------##
    #  업샘플링 과정에서 사용되는 convolution layer의 적층
    ##-------------------------------------------------------------------------------##
    u1 = deconv2d(d7, d6, self.gf*8)
    u2 = deconv2d(u1, d5, self.gf*8)
    u3 = deconv2d(u2, d4, self.gf*8)
    u4 = deconv2d(u3, d3, self.gf*4)
    u5 = deconv2d(u4, d2, self.gf*2)
    u6 = deconv2d(u5, d1, self.gf)

    u7 = UpSampling2D(size=2)(u6)

    output_img = Conv2D(self.channels, kernel_size=4, strides=1, padding='same', activation='tanh')(u7)

    return Model(d0, output_img)

  ##=================================================================================##
  #  판별자 함수 정의
  ##=================================================================================##
  def build_discriminator(self):

    ##-------------------------------------------------------------------------------##
    #  판별자 레이어 정의
    #  - 사용함수 :  LeakyReLU
    ##-------------------------------------------------------------------------------##
    def d_layer(layer_input, filters, kernel_size=4, bn=True):
      d = Conv2D(filters, kernel_size=kernel_size, strides=2, padding='same')(layer_input)
      d = LeakyReLU(alpha=0.2)(d)
      if bn:
        d = BatchNormalization(momentum=0.8)(d)
      return d

    img_A = Input(shape=self.img_shape)
    img_B = Input(shape=self.img_shape)


    ##-------------------------------------------------------------------------------##
    #  combined_imgs 입력
    ##-------------------------------------------------------------------------------##
    combined_imgs = Concatenate(axis=-1)([img_A, img_B])

    ##-------------------------------------------------------------------------------##
    #  판별자 레이어 적층 모양
    ##-------------------------------------------------------------------------------##
    d1 = d_layer(combined_imgs, self.df, bn=False)
    d2 = d_layer(d1, self.df*1)
    d3 = d_layer(d2, self.df*2)
    d4 = d_layer(d3, self.df*4)

    validity = Conv2D(1, kernel_size=4, strides=1, padding='same')(d4)

    return Model([img_A, img_B], validity)

  ##=================================================================================##
  #  훈련(Train) 함수 정의
  ##=================================================================================##
  def train(self, epochs, batch_size=1, sample_interval=50):
    start_time = datetime.datetime.now()

    # adversarial loss ground truths
    valid = np.ones((batch_size,) + self.disc_patch)
    fake = np.zeros((batch_size,) + self.disc_patch)

    for epoch in range(epochs):
      for batch_i, (imgs_A, imgs_B) in enumerate(self.data_loader.load_batch(batch_size)):

        ##---------------------------------------------------------------------------##
        #  판별자 학습
        ##---------------------------------------------------------------------------##
        fake_B = self.generator.predict(imgs_A)

        if np.random.random() < 0.5:
          d_loss = self.discriminator.train_on_batch([imgs_A, imgs_B], valid)
        else:
          d_loss = self.discriminator.train_on_batch([imgs_A, fake_B], fake)
        # d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

        ##---------------------------------------------------------------------------##
        #  생성자 학습
        ##---------------------------------------------------------------------------##
        g_loss = self.combined.train_on_batch([imgs_A, imgs_B], [valid, imgs_B])

        ##---------------------------------------------------------------------------##
        #  학습과정 화면에 출력
        ##---------------------------------------------------------------------------##
        elapsed_time = datetime.datetime.now() - start_time

        print('[Epoch %d/%d] [Batch %d/%d] [D loss: %f, acc: %3d%%] [G loss: %f] time: %s' % (epoch, epochs, batch_i, self.data_loader.n_batches, d_loss[0], 100*d_loss[1], g_loss[0], elapsed_time))

        ##---------------------------------------------------------------------------##
        #  일정 과정마다 샘플 이미지 저장
        ##---------------------------------------------------------------------------##
        if batch_i % sample_interval == 0:
          self.sample_images(epoch, batch_i, d_loss)

        ##---------------------------------------------------------------------------##
        #  판별자가 낮은 정확성을 가질 때 샘플 이미지 저장
        ##---------------------------------------------------------------------------##
        if epoch > 9 and d_loss[1] < 0.6:
          self.sample_images(epoch, batch_i, d_loss, low=True)

  ##=================================================================================##
  #  샘플 이미지 저장함수 정
  ##=================================================================================##
  def sample_images(self, epoch, batch_i, d_loss, low=False):
    os.makedirs('samples', exist_ok=True)

    imgs_A, imgs_B = self.data_loader.load_data(batch_size=3, is_train=False)
    fake_B = self.generator.predict(imgs_A)

    gen_imgs = np.concatenate([imgs_A, fake_B, imgs_B])

    # rescale images to 0 - 1
    gen_imgs = 0.5 * gen_imgs + 0.5

    titles = ['Input', 'Generated', 'Ground Truth']
    fig, axs = plt.subplots(3, 3)

    for i in range(3):
      for j in range(3):
        axs[i, j].imshow(gen_imgs[3*i+j].squeeze(), cmap='gray')
        axs[i, j].set_title(titles[i])
        axs[i, j].axis('off')

    if low:
      fig.savefig('samples/low_%d_%d_%d.png' % (epoch, batch_i, d_loss[1] * 100))
    else:
      fig.savefig('samples/%d_%d_%d.png' % (epoch, batch_i, d_loss[1] * 100))

    plt.close()

if __name__ == '__main__':
  gan = Pix2Pix()
  gan.train(epochs=500, batch_size=1, sample_interval=1000)
