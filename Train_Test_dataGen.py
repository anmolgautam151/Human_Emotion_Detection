import pandas as pd
import numpy as np
import os
import cv2


# data = pd.read_csv("./data/train.csv")
# print(data.head())
emotions = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]
imagecount = [0,0,0,0,0,0,0]

data = pd.read_csv("./data/icml_face_data.csv")
print(data.head())

for i,j,k in zip(data["emotion"], data[" Usage"],data[" pixels"]):
    pixel = []
    pixels = k.split(' ')
    for x in pixels:
        pixel.append(float(x))
    pixel = np.array(pixel)
    image = pixel.reshape(48,48)

    if(j == "Training"):
        if not os.path.exists("./train/"+emotions[i]):
            os.mkdir("./train/"+emotions[i])

        path = "./train/"+emotions[i]+"/"+str(imagecount[i])+".jpg"
        imagecount[i] = imagecount[i]+1

    else:
        if not os.path.exists("./validation/" + emotions[i]):
            os.mkdir("./validation/" + emotions[i])

        path = "./validation/" + emotions[i] + "/" + str(imagecount[i]) + ".jpg"
        imagecount[i] = imagecount[i] + 1
    cv2.imwrite(path,image)



# for i,j in zip(data["emotion"], data["pixels"]):
#     pixel = []
#     pixels = j.split(' ')
#     for k in pixels:
#         pixel.append(float(k))
#     pixel = np.array(pixel)
#     image = pixel.reshape(48,48)
#
#     if not os.path.exists("./train/"+emotions[i]):
#         os.mkdir("./train/"+emotions[i])
#
#     path = "./train/"+emotions[i]+"/"+str(imagecount[i])+".jpg"
#     imagecount[i] = imagecount[i]+1
#
#     cv2.imwrite(path,image)
#
# data = pd.read_csv("./data/test.csv")
# print(data.head())
# count = 0
#
# for j in zip(data["pixels"]):
#     pixel = []
#     pixels = j[0].split(' ')
#     for k in pixels:
#         pixel.append(float(k))
#     pixel = np.array(pixel)
#     image = pixel.reshape(48,48)
#
#     if not os.path.exists("./test/"):
#         os.mkdir("./test/")
#
#     path = "./test/"+str(count)+".jpg"
#     count+=1
#
#     cv2.imwrite(path,image)
