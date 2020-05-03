

import pandas as pd
from matplotlib.image import imread
import matplotlib.pyplot as plt
import numpy as np
import os

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential,load_model,save_model

from tensorflow.keras.layers import Conv2D,Dropout,Flatten,Dense,MaxPool2D

from tensorflow.keras.callbacks import EarlyStopping

from tensorflow.keras.preprocessing import image


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def model_creation():
    train_path='C://Users/Asus/Desktop/Datasets/mask_dataset/training_set'
    test_path='C://Users/Asus/Desktop/Datasets/mask_dataset/test_set'
    print(os.listdir(train_path+'/mask'))

    mask_image=imread(train_path+'/mask/13.png')


    image_shape=mask_image.shape

    print(image_shape,mask_image.ndim)



    train_datagen=ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.3,
        horizontal_flip=True)

    test_datagen=ImageDataGenerator(shear_range=0.2,
        zoom_range=0.3,
        horizontal_flip=True
                 )

    traingen=train_datagen.flow_from_directory(

            train_path,
            target_size=image_shape[:2],
            batch_size=10,
            color_mode='rgb',
            class_mode='binary'

    )

    test_gen = test_datagen.flow_from_directory(
        test_path,
        target_size=image_shape[:2],
        batch_size=10,
        class_mode='binary',
        color_mode='rgb'
    )




    model=Sequential()

    model.add(Conv2D(filters=32,kernel_size=(3,3),input_shape=image_shape,activation='relu'))
    model.add(MaxPool2D(pool_size=(2,2)))

    model.add(Conv2D(filters=64,kernel_size=(3,3),input_shape=image_shape,activation='relu'))
    model.add(MaxPool2D(pool_size=(2,2)))

    model.add(Conv2D(filters=128,kernel_size=(3,3),input_shape=image_shape,activation='relu'))
    model.add(MaxPool2D(pool_size=(2,2)))


    model.add(Flatten())


    model.add(Dropout(0.5))

    model.add(Dense(128,activation='relu'))
    #o/p layer

    model.add(Dense(1,activation='sigmoid'))


    model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])



    earlystop=EarlyStopping(monitor='val_loss',patience=10)

    #print(model.summary())


    #model.fit_generator(traingen,validation_data=test_gen,epochs=5,callbacks=[earlystop])




    #model.save('Mask_Classifier.h5')

    model1=load_model('Mask_Classifier.h5')

    test_img=imread('tests.jpg')
    plt.imshow(test_img)
    myimage=image.load_img('tests.jpg',target_size=image_shape)
    myimage_arr=image.img_to_array(myimage)
    print(myimage_arr.shape)
    myimage_arr=np.expand_dims(myimage_arr,axis=0)
    print(myimage_arr.shape)

    pred=model1.predict_classes(myimage_arr)[0]

    print(pred)

    li=['mask','non-mask']

    print(li[pred[0]])
    plt.show()
pass




if __name__ == '__main__':
    model_creation()


