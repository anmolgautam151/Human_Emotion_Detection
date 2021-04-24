import os

train_counts = []
path = "./train/"
train_dir = os.listdir(path)
train_name = []

for folder in train_dir:
    class_path = path + folder +"/"
    list_train = []
    count = 0

    for file in os.listdir(class_path):
        count+=1

    train_name.append(folder)
    train_counts.append(count)

print(train_name)
print(train_counts)