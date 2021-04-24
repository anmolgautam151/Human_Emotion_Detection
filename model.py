from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, BatchNormalization, Dense, Flatten
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.callbacks import TensorBoard

num_classes = 6

Img_Height = 48
Img_width = 48

batch_size = 32

train_dir = "./train/"
validation_dir = "./validation/"

train_datagen = ImageDataGenerator(rescale=1./255,
                                   rotation_range=60,
                                   shear_range=0.5,
                                   zoom_range=0.5,
                                   width_shift_range=0.5,
                                   height_shift_range=0.5,
                                   horizontal_flip=True,
                                   fill_mode='nearest')

validation_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(train_dir,
                                                    color_mode='grayscale',
                                                    target_size=(Img_Height, Img_width),
                                                    batch_size=batch_size,
                                                    class_mode='categorical',
                                                    shuffle=True)

validation_generator = validation_datagen.flow_from_directory(validation_dir,
                                                              color_mode='grayscale',
                                                              target_size=(Img_Height, Img_width),
                                                              batch_size=batch_size,
                                                              class_mode='categorical',
                                                              shuffle=True)


model = Sequential()

# Block-1: The First Convolutional Block

model.add(Conv2D(filters=32, kernel_size=(3,3), padding='same',
                 kernel_initializer='he_normal',
                 activation="relu",
                 input_shape=(Img_Height, Img_width, 1),
                 name="Conv1"))

model.add(BatchNormalization(name="Batch_Norm1"))

model.add(Conv2D(filters=32, kernel_size=(3,3), padding='same',
                 kernel_initializer='he_normal',
                 activation="relu", name="Conv2"))

model.add(BatchNormalization(name="Batch_Norm2"))
model.add(MaxPooling2D(pool_size=(2,2), name="Maxpool1"))
model.add(Dropout(0.2, name="Dropout1"))

# Block-2: The Second Convolutional Block

model.add(Conv2D(filters=64, kernel_size=(3,3), padding='same',
                 kernel_initializer='he_normal',
                 activation="relu", name="Conv3"))

model.add(BatchNormalization(name="Batch_Norm3"))

model.add(Conv2D(filters=64,kernel_size=(3,3),padding='same',
                 kernel_initializer='he_normal',
                 activation="relu", name="Conv4"))

model.add(BatchNormalization(name="Batch_Norm4"))
model.add(MaxPooling2D(pool_size=(2,2), name="Maxpool2"))
model.add(Dropout(0.2, name="Dropout2"))

# Block-3: The Third Convolutional Block

model.add(Conv2D(filters=128, kernel_size=(3,3), padding='same',
                 kernel_initializer='he_normal',
                 activation="relu", name="Conv5"))

model.add(BatchNormalization(name="Batch_Norm5"))

model.add(Conv2D(filters=128, kernel_size=(3,3), padding='same',
                 kernel_initializer='he_normal',
                 activation="relu", name="Conv6"))

model.add(BatchNormalization(name="Batch_Norm6"))
model.add(MaxPooling2D(pool_size=(2,2), name="Maxpool3"))
model.add(Dropout(0.2, name="Dropout3"))

# Block-4: The Fully Connected Block

model.add(Flatten(name="Flatten"))
model.add(Dense(64, activation="relu", kernel_initializer='he_normal', name="Dense"))
model.add(BatchNormalization(name="Batch_Norm7"))
model.add(Dropout(0.5, name="Dropout4"))

# Block-5: The Output Block

model.add(Dense(num_classes, activation="softmax", kernel_initializer='he_normal', name = "Output"))
#
# ####
checkpoint = ModelCheckpoint("emotions.h5", monitor='accuracy', verbose=1,
                              save_best_only=True, mode='auto', period=1)

reduce = ReduceLROnPlateau(monitor='accuracy', factor=0.2, patience=10,
                           min_lr=0.0001, verbose = 1)

logdir='logs'
tensorboard_Visualization = TensorBoard(log_dir=logdir, histogram_freq=False)
#
# ####compile
#
#
train_samples = 28273
validation_samples = 7067
epochs = 150
batch_size = 64

model.compile(loss='categorical_crossentropy',
              optimizer=Adam(lr=0.001),
              metrics=['accuracy'])

model.fit(train_generator,
          steps_per_epoch=train_samples // batch_size,
          epochs=epochs,
          callbacks=[checkpoint, reduce, tensorboard_Visualization],
          validation_data=validation_generator,
          validation_steps=validation_samples // batch_size,
          shuffle=True)