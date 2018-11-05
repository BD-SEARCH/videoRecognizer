# STEP3. 트레이닝 및 정확도 검사

import cv2 as cv
import os
from keras.applications import VGG16
from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout, Activation
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras import models
from keras import layers
from keras import optimizers
from keras import backend as K

img_w, img_h = 100, 100
train_dir = './dataset_img/training_set'
test_dir = './dataset_img/test_set'
# train_samples = 550
train_samples = sum([len(files) for r, d, files in os.walk('./dataset_img/training_set')])
# test_samples = 100
test_samples = sum([len(files) for r, d, files in os.walk('./dataset_img/test_set')])
epochs = 20
train_batch_size = 10
test_batch_size = 5
classnum = len(os.listdir(train_dir))-1

print(train_samples)
print(test_samples)
print(classnum)

if K.image_data_format() == 'channels_first':
    input_shape = (classnum, img_w, img_h)
else:
    input_shape = (img_w, img_h, classnum)

# 1. create the model
vgg_conv = VGG16(weights='imagenet', include_top=False, input_shape=(img_w, img_h, 3))
# for layer in vgg_conv.layers[:-4]: layer.trainable = False
# for layer in vgg_conv.layers:
#     print(layer, layer.trainable)
for layer in vgg_conv.layers: layer.trainable = False

model = Sequential()
model.add(vgg_conv)

model.add(layers.Flatten())
model.add(layers.Dense(1024, activation='relu'))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(classnum, activation='sigmoid'))

# Show a summary of the model. Check the number of trainable parameters
# model.summary()

# 2. create dataset
train_datagen = ImageDataGenerator(rescale=1./255)
# train_datagen = ImageDataGenerator(rescale=1./255,
#                                   rotation_range=15,
#                                   width_shift_range=0.1,
#                                   height_shift_range=0.1,
#                                   # shear_range=0.5,
#                                   # zoom_range=[0.8, 2.0],
#                                   horizontal_flip=True,
#                                 #   vertical_flip=True,
#                                   fill_mode='nearest')
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size = (img_w, img_h),
    batch_size = train_batch_size,
    class_mode = 'categorical'
)

test_datagen = ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size = (img_w, img_h),
    batch_size = test_batch_size,
    class_mode = 'categorical',
    shuffle = False
)

# 3. set the model learning
# optimizer=adam
model.compile(loss='categorical_crossentropy', optimizer=optimizers.RMSprop(lr=1e-4), metrics=['acc'])

# 4. learn the model
model.fit_generator(train_generator,
                    # steps_per_epoch = train_samples
                    steps_per_epoch=train_generator.samples/train_generator.batch_size,
                    epochs=epochs,
                    validation_data = test_generator,
                    # validation_steps = test_batch_size
                    validation_steps=test_generator.samples/test_generator.batch_size,
                    verbose=1)
# model.save_weights('test2.h5')
model.save('test.h5')

# 5. evaluate the model
print("-- Evaluate(정확도) --")
scores = model.evaluate_generator(test_generator, steps=5)
print("%s: %.2f%%" %(model.metrics_names[1], scores[1]*100))

# 모든 자원을 해제
# cap.release()
# cv.destroyAllWindows()

# STEP4. 정답 틀린 이미지 보여주기

import numpy as np
import matplotlib.pyplot as plt

# Get the filenames from the generator
fnames = test_generator.filenames

# Get the ground truth from generator
ground_truth = test_generator.classes

# Get the label to class mapping from the generator
label2index = test_generator.class_indices

# Getting the mapping from class index to class label
idx2label = dict((v,k) for k,v in label2index.items())

# Get the predictions from the model using the generator
predictions = model.predict_generator(test_generator, steps=test_generator.samples/test_generator.batch_size,verbose=1)
predicted_classes = np.argmax(predictions,axis=1)

errors = np.where(predicted_classes != ground_truth)[0]
print("No of errors = {}/{}".format(len(errors),test_generator.samples))

# Show the errors
for i in range(len(errors)):
    pred_class = np.argmax(predictions[errors[i]])
    pred_label = idx2label[pred_class]

    title = 'Original label:{}, Prediction :{}, confidence : {:.3f}'.format(
        fnames[errors[i]].split('/')[0],
        pred_label,
        predictions[errors[i]][pred_class])

    original = load_img('{}/{}'.format(test_dir,fnames[errors[i]]))
    plt.figure(figsize=[7,7])
    plt.axis('off')
    plt.title(title)
    plt.imshow(original)
    plt.show()

# STEP6. guess
# 0. 사용할 패키지 불러오기
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

# 1. data 준비
test_datagen = ImageDataGenerator(rescale=1./255)
# 검증용 generator 생성
test_generator = test_datagen.flow_from_directory(
        './dataset_img/test_set',
        target_size=(img_w, img_h),
        batch_size=1,
        class_mode='categorical')

# 2. call model
from keras.models import load_model
# model = load_weights('test2.h5')
model = load_model('test.h5')

# 3. use model
print("-- Predict --")
output = model.predict_generator(test_generator, steps=5)
np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})

# print("//")
# print(test_generator.class_indices)
# print(output)
# print("//")
class_list = list(test_generator.class_indices.keys())
res_list = [0]*len(test_generator.class_indices)
for res in output:
    tmp = list(map(float,str(res)[1:len(str(res))-1].split()))
    max_index = tmp.index(max(tmp))
    res_list[max_index]+=1
print(res_list)
print(max(res_list)/sum(res_list)*100,"%의 확률로",class_list[res_list.index(max(res_list))])
# print(class_list[res_list.index(max(res_list))])
# print("ans:",class_list(res_list.index(max(res_list))))
# print(test_generator.filenames)
