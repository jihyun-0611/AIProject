from PIL import Image
import os, glob
import numpy as np
from sklearn.model_selection import train_test_split

interest_dir = "./image"
categories = ["cafe", "fishing", "golf", "pet"]
nb_classes = len(categories)

image_width = 64
image_height = 64
pixels = image_width* image_height * 3

X = []
Y = []
for idx, cat in enumerate(categories):
    label = [0 for i in range(nb_classes)]
    label[idx] = 1

    image_dir = interest_dir + "/" + cat
    files = glob.glob(image_dir + "/*.jpg")
    for i, f in enumerate(files):
        img = Image.open(f)
        img = img.convert("RGB")
        img = img.resize((image_width, image_height))
        data = np.asarray(img)  # numpy 배열로 변환
        X.append(data)
        Y.append(label)
        if i % 10 == 0:
            print(i, "\n", data)

X = np.array(X)
Y = np.array(Y)

X_train, X_test, y_train, y_test = train_test_split(X, Y)
xy = (X_train, X_test, y_train, y_test);
np.save("pre_img.npy", xy)
print("ok",len(Y))