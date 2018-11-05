# STEP5. guess
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
