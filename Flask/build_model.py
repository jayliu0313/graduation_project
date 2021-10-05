import os

# os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import tensorflow as tf
from tensorflow.keras import Sequential,Input
from tensorflow.keras.layers import Conv2D,MaxPooling2D,Flatten,Dense,Activation,BatchNormalization,GaussianNoise


def load_model():
    print("load model~~~~~~~~~~~~~~~~~~~~~~")
    print(os.getcwd())
    # WEIGHTS_PATH_2 = r"flower_vgg"
    WEIGHTS_PATH_2 = "weights/2021_95ACC"#vgg_2020_4_29_night2"
    global model
    

    #filepath_2 =get_file('vgg_2020_12_7.h5',WEIGHTS_PATH_2,cache_subdir='models')
    ######################
    
    model=Sequential()
    
    # model.add(GaussianNoise(stddev=0.001, input_shape=[224 ,224 ,3]))
    
    model.add(Conv2D(input_shape=[224 ,224 ,3],name ="block1_conv1",filters=64,kernel_size=(3,3),padding='same'))
    # model.add(Conv2D(name ="block1_conv1",filters=64,kernel_size=(3,3),padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    
    model.add(Conv2D(name ="block1_conv2",filters=64,kernel_size=(3,3),padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    
    model.add(MaxPooling2D(name="block1_pool",pool_size=(2,2),strides=(2,2)))
    ###################2
    model.add(Conv2D(name ="block2_conv1",filters=128,kernel_size=(3,3),padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    
    model.add(Conv2D(name ="block2_conv2",filters=128,kernel_size=(3,3),padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    
    model.add(MaxPooling2D(name="block2_pool",pool_size=(2,2),strides=(2,2)))
    ###################3
    model.add(Conv2D(name ="block3_conv1",filters=256,kernel_size=(3,3),padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    
    model.add(Conv2D(name ="block3_conv2",filters=256,kernel_size=(3,3),padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    
    model.add(Conv2D(name ="block3_conv3",filters=256,kernel_size=(3,3),padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    
    model.add(Conv2D(name ="block3_conv4",filters=256,kernel_size=(3,3),padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    
    model.add(MaxPooling2D(name="block3_pool",pool_size=(2,2),strides=(2,2)))
    ###################4
    model.add(Conv2D(name ="block4_conv1",filters=512,kernel_size=(3,3),padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    
    model.add(Conv2D(name ="block4_conv2",filters=512,kernel_size=(3,3),padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    
    model.add(Conv2D(name ="block4_conv3",filters=512,kernel_size=(3,3),padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    
    model.add(Conv2D(name ="block4_conv4",filters=512,kernel_size=(3,3),padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    
    model.add(MaxPooling2D(name="block4_pool",pool_size=(2,2),strides=(2,2)))
    
    ###################4
    model.add(Conv2D(name ="block5_conv1",filters=512,kernel_size=(3,3),padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    
    model.add(Conv2D(name ="block5_conv2",filters=512,kernel_size=(3,3),padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    
    model.add(Conv2D(name ="block5_conv3",filters=512,kernel_size=(3,3),padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    
    model.add(Conv2D(name ="block5_conv4",filters=512,kernel_size=(3,3),padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    
    model.add(MaxPooling2D(name="block5_pool",pool_size=(2,2),strides=(2,2)))
    
    # model.add(Flatten())
    
    # model.add(Dense(4096))
    # model.add(BatchNormalization())
    # model.add(Activation('relu'))
    # model.add(Dense(1000))
    # model.add(BatchNormalization())
    # model.add(Activation('relu'))
    model.add(BatchNormalization())
    # model.add(Activation('relu'))

    model.add(Flatten())
    model.add(Dense(256))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dense(128))
    model.add(BatchNormalization())
    model.add(Activation('relu')) 

    model.add(Dense(101))
    
    model.add(Activation('softmax'))
    
    model.load_weights(WEIGHTS_PATH_2)#,by_name=True )
    
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    #global graph
    #model = ResNet50(weights='imagenet')
    # 初始化 tensorflow graph
    #graph = tf.get_default_graph()
    return model


model = load_model()