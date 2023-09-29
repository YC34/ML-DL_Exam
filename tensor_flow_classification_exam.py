#!/usr/bin/env python
# coding: utf-8

# In[83]:


## 텐서 플로우 예제 파일로 실습하기.


# In[84]:


### 텐셔 플로우 버전 확인.

import tensorflow as tf
print(tf.__version__)


# In[85]:


## 이미지의 형태를 어떻게 변환하였는지 체크하기 위해 각각의 차원을 확인 


# In[86]:


### 텐서 플로우 샘플 이미지 파일 가져오기.

import numpy as np
import matplotlib.pyplot as plt


fashion_mnist = tf.keras.datasets.fashion_mnist
# 데이터의 타입 확인 (tuple)
print(type(fashion_mnist.load_data()))
# 데이터 확인
print("tuple의 수 : " ,len(fashion_mnist.load_data()))
# print(fashion_mnist.load_data()[0])
# print(fashion_mnist.load_data()[1])


print(len(fashion_mnist.load_data()[0][0].shape))
print(len(fashion_mnist.load_data()[0][1].shape))
# print(fashion_mnist.load_data()[0][0][0])
# print(fashion_mnist.load_data()[0][0][1])


# print(len(fashion_mnist.load_data()[1][0]))
# print(len(fashion_mnist.load_data()[1][1]))
# print(len(fashion_mnist.load_data()[1][0][0]))
# print(len(fashion_mnist.load_data()[1][0][1]))

# 3차원 배열로 , 총 7만개의 이미지가 존재한다. tuple(0)은 train을 위한 6만개 2차원 배열이 28*28픽셀안에 각각의 이미지가 표현된다.
# 이 3차원 배열에는 6만개의 label을 갖는 1차원 배열이 있다. 
# tuple[1]은 test를 위한 데이터 같은 형태이지만, 총 1만개의 이미지가 존재한다.
# 각각의 역할로 구분해준다.

# 튜플이 2개인 형태이므로,
# ( , ) ,( , ) = data 라는 형태로 변형해준다. 첫튜플은 train을 위한, 두번째 튜플은 test용 

(train_data ,train_label) ,(test_data ,test_label ) = fashion_mnist.load_data()


print(train_data.shape)
print(train_label.shape)
print(train_data[0])
print(train_label[0])


# In[87]:


## 텐셔플로우에서 제공하는 정보를 이용하여, 각각의 레이블에 대한 클래스 명시를 해준다


# In[88]:


# 하는 이유는 총 6만개의 train 데이터는 각각의 label이 명시되어있다. 
# ex) train[0] 28*28로 된 2차원 배열로 되어있는데, 그에 맞는 label은 9로 되어있다. 
# 0	T-shirt/top
# 1 	Trouser
# 2	Pullover
# 3	Dress
# 4	Coat
# 5	Sandal
# 6	Shirt
# 7	Sneaker
# 8	Bag
# 9	Ankle boot
# 각각의 분류 label에 대한 명시를 해준다. type은 list로 하여, 순서대로 클래스를 지정한다.
# 각각의 데이터가 label이 없다면, 원 핫 인코딩을 진행하여 분류한다.


class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# 실질적인 학습시킬 train_data와 train_data를 분류하는 train_label의 타입과형태를 확인한다.

print(type(train_data))
print(train_data.shape)

print(type(train_label))
print(train_label.shape)

# 학습시킨 후 test시킬 데이터도 확인.

print(type(test_data))
print(test_data.shape)

print(type(test_label))
print(test_label.shape)

# label의 갯수와 2차원 배열의 갯수가 동일하다.


# In[89]:


# 각각의 데이터가 실제로 정확한지 시각화 한다.

plt.figure() # 캔버스 초기화
plt.imshow(train_data[0])
plt.colorbar()
plt.grid(False)
plt.show()

print(train_label[0])
# label과 2차원 배열의 이미지가 일치한다. 


# In[90]:


## 데이터 정규화. 
## 여러가지 방법이 있는데 이 sample에서는 모든 데이터를 0~1사이로 정규화하였다.
## 정규화 하는 이유는 , 모든 데이터들을 동일 한 분산으로 놓기 위해..? 동일한 형상을 만들기 위해???. 
## 0~255라는 숫자안에 모든 숫자들이 담겨 있으므로, 가장큰 255.0으로 나누면 1보다는 작고 0보다는 크보다 작은 숫자가 나온다.
## 일반화를 위함.

print(train_data[0])

preprocess_train_data= train_data /255.0
preprocess_test_data =test_data /255.0

print(preprocess_train_data[0])


# In[37]:


## 정규화 된 데이터를 그려보자.


# In[91]:


plt.figure(figsize=(10,10)) # 캔버스 초기화 및 사이즈 조절.
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(preprocess_train_data[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[train_label[i]])
plt.show()


# In[92]:


## 학습시킬 모델 생성 (tensorflow를 이용)


# In[105]:


## 모델 생성,Flatten 28*28의 2차원을 1차원으로 펼치는 함수..? >> 입력층?
## Dense 입력층은 128개, 활성화 함수는 relu를 사용. >>> 은닉층? 
## 10개의 출력층으로 만든다? 옵션을 선택하지 않았으므로, 순수한 선형을 출력. 10개의 출력층으로 한 이유는 현재 분류한 클래스의 종류가 총 10개이기 떄문
## 0~9까지의 클래스이므로 출력층은 10개로 한다. 
## Sequential 모델은 일렬로 층을 쌓은 네트워크를 빠르게 만들 떄 사용하기 좋음.
## MLP(multilayer perception) - 인접한 모든 층이 완전히 연결된 신경망.    
## 각 유닛(뉴런)에게 어떤 특성을 찾아야 한다고 알려줄 필요가 없다 (특성공학이 필요가 없다. 생성된 모델들을 가지고 판단하기 때문)
## 각각의 유닛(뉴런)들은 자신들이 가지고 탐색해야할? 특성을 탐색한다.. 이런 여러가지의 특성들을 학습하여 판별한다. ==> 은닉층? 
## ex) 유닛(뉴런)A는 입력 픽셀의 개별 채널에 대한 값을 받습니다.(Flatten??)
##     유닛(뉴런)B는 입력값을 결합하여 에지와 같이 측정한 저수준 특성이 존재할 떄 가장 큰 값을 출력
##     유닛(뉴런)C는 저수준 특성을 결합하여 이미지에 치아와 같은 고수준 특성이 보일떄 가장 큰 값을 출력
##     유닛(뉴런)D는 고수준 특성을 결합하여 원본 이미지에 있는 클래스를 판별합니다..? ....이게 약 128개를 만들어 낸다? 
##     위의 은닉층들을 토대로 10개의 특성으로 구분하여 출력한다..? 
##     신경망 값은 입력의 절대값이 1보다 작을떄 가장 잘 작동한다. 그래서 전처리를 통해 0~1사이로 정규화를 해주어야한다. 0~!사이로 정규화한 근거.





# model = tf.keras.Sequential([
#     tf.keras.layers.Flatten(input_shape=(28, 28)),
#     tf.keras.layers.Dense(128, activation='relu'),
#     tf.keras.layers.Dense(10)
# ])


## 위의 예제 파일은 Sequential이용하여 만든 모델 

## 함수형 API를 사용하여 MLP모델 생성

from tensorflow.keras import layers,models

## 입력층 생성 (input)
input_layer = layers.Input(shape=(28,28))

## Flatten층 
x= layers.Flatten()(input_layer)
## Dense층 (유닛{뉴런?})
x = layers.Dense(units=128,activation='relu')(x)
# ## Dense층 하나 더 늘려보기
# x = layers.Dense(units=56,activation='relu')(x)


## 출력층 
## 활성함수는 지정해주지 않았으므로, 일반적인 선형으로 출력한다.?
output_layer = layers.Dense(units=10)(x)

model= models.Model(input_layer,output_layer)

model.summary()


## 입력층은 train_x ,즉 실제 28*28의 수와 맞아아햐며 , 출력층은 x_label 즉 클래스로 분류한 갯수와 맞아야한다. (10개의 클래스로 분류하였으므로 , 10이 되어야한다.)


# In[102]:


## 모델 compile

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])


# In[103]:


## 만들어둔 데이터 학습 진행

model.fit(preprocess_train_data, train_label, epochs=10)


# In[70]:


## test데이터.

test_loss, test_acc = model.evaluate(test_data,  test_label, verbose=2)

print('\nTest accuracy:', test_acc)


# In[71]:


## 예측 모델 생성? 

probability_model = tf.keras.Sequential([model, 
                                         tf.keras.layers.Softmax()])


# In[72]:


predictions = probability_model.predict(test_data)


# In[73]:


predictions[0]

