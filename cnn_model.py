from PIL import Image
import os, glob
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Activation, Dropout, Flatten, Dense
from tensorflow.keras.optimizers import Adam

# interest_dir = "./image"
# categories = ["cafe", "fishing", "golf", "pet"]
# nb_classes = len(categories)
#
# image_w = 64
# image_h = 64
# pixels = image_w * image_h * 3
#
# X = []
# Y = []
# for idx, cat in enumerate(categories):
#     label = [0 for i in range(nb_classes)]
#     label[idx] = 1
#
#     image_dir = interest_dir + "/" + cat
#     files = glob.glob(image_dir + "/*.jpg")
#     for i, f in enumerate(files):
#         img = Image.open(f)
#         img = img.convert("RGB")
#         img = img.resize((image_w, image_h))
#         data = np.asarray(img)  # numpy 배열로 변환
#         X.append(data)
#         Y.append(label)
#         if i % 10 == 0:
#             print(i, "\n", data)
#
# X = np.array(X)
# Y = np.array(Y)
#
# X_train, X_test, y_train, y_test = train_test_split(X, Y)

categories = ["cafe", "fishing", "golf", "pet"]
nb_classes = len(categories)

image_w = 64
image_h = 64

x_train, x_test, y_train, y_test = np.load("pre_img.npy")
x_train = x_train.astype(np.float32) / 255.0
x_test = x_test.astype(np.float32) / 255.0

print('X_train shape: ', x_train.shape)

# 모델 설계

model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=x_train.shape[1:], padding='same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3, 3), padding='same'))
model.add(Activation('relu'))

model.add(Conv2D(64, (3, 3)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())  # 벡터형태로 reshape
model.add(Dense(512))  # 출력
model.add(Activation('relu'))
model.add(Dropout(0.5))

model.add(Dense(nb_classes))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy',  # 최적화 함수 지정
              optimizer=Adam(),
              metrics=['accuracy'])
hist = model.fit(x_train, y_train, batch_size=32, epochs=10, validation_data=(x_test,y_test),verbose=2)
# 학습 완료된 모델 저장
# hdf5_file = "./image/model.hdf5"
# if os.path.exists(hdf5_file):
#     # 기존에 학습된 모델 불러들이기
#     model.load_weights(hdf5_file)
# else:
#     # 학습한 모델이 없으면 파일로 저장
#     model.fit(X_train, y_train, batch_size=32, epochs=10)
#     model.save_weights(hdf5_file)

score = model.evaluate(x_test, y_test, verbose=2)
print('loss=', score[0])  # loss
print('accuracy=', score[1])  # acc

import matplotlib.pyplot as plt

#정확률 곡선
plt.plot(hist.history['accuracy'])
plt.plot(hist.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.grid()
plt.show()

#손실함수 곡선
plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper right')
plt.grid()
plt.show()
