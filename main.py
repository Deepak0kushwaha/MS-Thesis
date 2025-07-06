#IMPORT LIBRARIES AND SET DIRECTORY
import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import backend as backend
from tensorflow.keras.initializers import HeNormal
from tensorflow.keras.layers import Input, Conv2D, ReLU, ELU, Concatenate, BatchNormalization, AveragePooling2D, MaxPooling2D, Dropout, Flatten, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import LearningRateScheduler, TensorBoard, CSVLogger, EarlyStopping, ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from PIL import Image

data_dir =  '../2000RPM_pump/2000rpm_allCWTlog'  # Update with the actual path to your dataset

#LOAD DATA
def load_data(data_dir, target_size=(312, 312)):
    images = []
    labels = []
    for class_label in os.listdir(data_dir):
        class_dir = os.path.join(data_dir, class_label)
        if os.path.isdir(class_dir):
            for file in os.listdir(class_dir):
                if file.endswith('.png'):
                    image_path = os.path.join(class_dir, file)
                    img = Image.open(image_path)
                    img = img.resize(target_size)
                    if img.mode != 'RGB':
                        img = img.convert('RGB')
                    img_array = np.array(img) / 255.0
                    images.append(img_array)
                    labels.append(class_label)
    return np.array(images), np.array(labels)

#SPLIT DATA
from sklearn.model_selection import train_test_split
from collections import Counter

images, labels = load_data(data_dir)
num_classes = len(set(labels))
unique_classes = sorted(set(labels))

# First split into training and temp (combination of validation and test)
X_train, X_temp, y_train, y_temp = train_test_split(
    images, labels,
    test_size=0.3,
    random_state=42,
    stratify=labels  # Stratify based on labels
)

# Then split the temp into validation and test sets
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp,
    test_size=0.5,
    random_state=42,
    stratify=y_temp  # Stratify based on the temp labels
)

# Now, you can count the instances of each class again, as before
train_counts = Counter(y_train)
val_counts = Counter(y_val)
test_counts = Counter(y_test)

# Display the counts for each class
print("Training set distribution:")
for class_label in unique_classes:
    print(f"Class {class_label}: {train_counts[class_label]} images")

print("\nValidation set distribution:")
for class_label in unique_classes:
    print(f"Class {class_label}: {val_counts[class_label]} images")

print("\nTest set distribution:")
for class_label in unique_classes:
    print(f"Class {class_label}: {test_counts[class_label]} images")


from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical

# Assume y_train, y_val, y_test are already defined and contain your labels
label_encoder = LabelEncoder()
y_train_encoded = to_categorical(label_encoder.fit_transform(y_train), num_classes=num_classes)
y_val_encoded = to_categorical(label_encoder.transform(y_val), num_classes=num_classes)
y_test_encoded = to_categorical(label_encoder.transform(y_test), num_classes=num_classes)


#MODIEFIED INCEPTIONV3 MODEL
def conv2d_bn(x, filters, num_row, num_col, padding='same', strides=(1, 1), name=None):
    if name is not None:
        bn_name = name + '_bn'
        conv_name = name + '_conv'
    else:
        bn_name = None
        conv_name = None
    if backend.image_data_format() == 'channels_first':
        bn_axis = 1
    else:
        bn_axis = 3
    x = layers.Conv2D(
        filters, (num_row, num_col),
        strides=strides,
        padding=padding,
        use_bias=True,
        name=conv_name)(x)
    x = layers.BatchNormalization(axis=bn_axis, scale=False, name=bn_name)(x)
    x = layers.Activation('relu', name=name)(x)
    return x

def conv2d_bn1(x1, filters, num_row, num_col, padding='same', strides=(1, 1), name=None):
    if name is not None:
        bn_name = name + '_bn'
        conv_name = name + '_conv'
    else:
        bn_name = None
        conv_name = None
    if backend.image_data_format() == 'channels_first':
        bn_axis = 1
    else:
        bn_axis = 3
    x1 = layers.Conv2D(
        filters, (num_row, num_col),
        strides=strides,
        padding=padding,
        use_bias=True,
        name=conv_name)(x1)
    x1 = layers.BatchNormalization(axis=bn_axis, scale=True, center=True, name=bn_name)(x1)
    x1 = layers.Activation('elu', name=name)(x1)
    return x1



# Deep CNN Model Definition with custom architecture
def build_custom_inception_model(input_shape, num_classes):
    if backend.image_data_format() == 'channels_first':
        channel_axis = 1
    else:
        channel_axis = -1

    input_img = keras.Input(shape=input_shape)
    
    #Block1
    x = conv2d_bn(input_img, 32, 3, 3, strides=(2, 2), padding='valid')
    x = conv2d_bn(x, 32, 3, 3, padding='valid')
    x = conv2d_bn(x, 64, 3, 3)
    x = layers.MaxPooling2D((3, 3), strides=(2, 2))(x)
    x = conv2d_bn(x, 80, 1, 1, padding='valid')
    x = conv2d_bn(x, 192, 3, 3, padding='valid')
    x = layers.MaxPooling2D((3, 3), strides=(2, 2))(x)

    #Block2
    x1 = conv2d_bn1(input_img, 32, 3, 3, strides=(2, 2), padding='valid')
    x1 = conv2d_bn1(x1, 32, 3, 3, padding='valid')
    x1 = conv2d_bn1(x1, 64, 3, 3)
    x1 = layers.MaxPooling2D((3, 3), strides=(2, 2))(x1)
    x1 = conv2d_bn1(x1, 80, 1, 1, padding='valid')
    x1 = conv2d_bn1(x1, 192, 3, 3, padding='valid')
    x1 = layers.MaxPooling2D((3, 3), strides=(2, 2))(x1)


    # mixed 0 (relu): 35 x 35 x 256
    branch1x1 = conv2d_bn(x, 64, 1, 1)

    branch5x5 = conv2d_bn(x, 48, 1, 1)
    branch5x5 = conv2d_bn(branch5x5, 64, 5, 5)

    branch3x3dbl = conv2d_bn(x, 64, 1, 1)
    branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3)
    branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3)

    branch_pool = layers.AveragePooling2D((3, 3),
                                          strides=(1, 1),
                                          padding='same')(x)
    branch_pool = conv2d_bn(branch_pool, 32, 1, 1)


    # mixed 0 (elu): 35 x 35 x 256
    branch1x1_1 = conv2d_bn1(x1, 64, 1, 1)

    branch5x5_1 = conv2d_bn1(x1, 48, 1, 1)
    branch5x5_1 = conv2d_bn1(branch5x5_1, 64, 5, 5)

    branch3x3dbl_1 = conv2d_bn1(x1, 64, 1, 1)
    branch3x3dbl_1 = conv2d_bn1(branch3x3dbl_1, 96, 3, 3)
    branch3x3dbl_1 = conv2d_bn1(branch3x3dbl_1, 96, 3, 3)

    branch_pool_1 = layers.AveragePooling2D((3, 3),
                                          strides=(1, 1),
                                          padding='same')(x1)
    branch_pool_1 = conv2d_bn(branch_pool_1, 32, 1, 1)



    x2 = layers.concatenate(
        [branch1x1, branch5x5, branch3x3dbl, branch_pool, branch1x1_1, branch5x5_1, branch3x3dbl_1, branch_pool_1],
        axis=channel_axis,
        name='mixed0')

    # mixed 1 (relu): 35 x 35 x 288
    branch1x1 = conv2d_bn(x2, 64, 1, 1)

    branch5x5 = conv2d_bn(x2, 48, 1, 1)
    branch5x5 = conv2d_bn(branch5x5, 64, 5, 5)

    branch3x3dbl = conv2d_bn(x2, 64, 1, 1)
    branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3)
    branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3)

    branch_pool = layers.AveragePooling2D((3, 3),
                                          strides=(1, 1),
                                          padding='same')(x2)
    branch_pool = conv2d_bn(branch_pool, 64, 1, 1)


    # mixed 1 (elu): 35 x 35 x 288

    branch1x1_1 = conv2d_bn1(x2, 64, 1, 1)

    branch5x5_1 = conv2d_bn1(x2, 48, 1, 1)
    branch5x5_1 = conv2d_bn1(branch5x5_1, 64, 5, 5)

    branch3x3dbl_1 = conv2d_bn1(x2, 64, 1, 1)
    branch3x3dbl_1 = conv2d_bn1(branch3x3dbl_1, 96, 3, 3)
    branch3x3dbl_1 = conv2d_bn1(branch3x3dbl_1, 96, 3, 3)

    branch_pool_1 = layers.AveragePooling2D((3, 3),
                                          strides=(1, 1),
                                          padding='same')(x2)
    branch_pool_1 = conv2d_bn(branch_pool_1, 64, 1, 1)


    x3 = layers.concatenate([branch1x1, branch5x5, branch3x3dbl, branch_pool, branch1x1_1, branch5x5_1, branch3x3dbl_1, branch_pool_1],
        axis=channel_axis,
        name='mixed1')

    # mixed 2 (relu): 35 x 35 x 288
    branch1x1 = conv2d_bn(x3, 64, 1, 1)

    branch5x5 = conv2d_bn(x3, 48, 1, 1)
    branch5x5 = conv2d_bn(branch5x5, 64, 5, 5)

    branch3x3dbl = conv2d_bn(x3, 64, 1, 1)
    branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3)
    branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3)

    branch_pool = layers.AveragePooling2D((3, 3),
                                          strides=(1, 1),
                                          padding='same')(x3)
    branch_pool = conv2d_bn(branch_pool, 64, 1, 1)


    # mixed 2 (elu): 35 x 35 x 288
    branch1x1_1 = conv2d_bn1(x3, 64, 1, 1)

    branch5x5_1 = conv2d_bn1(x3, 48, 1, 1)
    branch5x5_1 = conv2d_bn1(branch5x5_1, 64, 5, 5)

    branch3x3dbl_1 = conv2d_bn1(x3, 64, 1, 1)
    branch3x3dbl_1 = conv2d_bn1(branch3x3dbl_1, 96, 3, 3)
    branch3x3dbl_1 = conv2d_bn1(branch3x3dbl_1, 96, 3, 3)

    branch_pool_1 = layers.AveragePooling2D((3, 3),
                                          strides=(1, 1),
                                          padding='same')(x3)
    branch_pool_1 = conv2d_bn1(branch_pool_1, 64, 1, 1)


    x4 = layers.concatenate(
        [branch1x1, branch5x5, branch3x3dbl, branch_pool, branch1x1_1, branch5x5_1, branch3x3dbl_1, branch_pool_1],
        axis=channel_axis,
        name='mixed2')

    # mixed 3 (relu): 17 x 17 x 768
    branch3x3 = conv2d_bn(x4, 384, 3, 3, strides=(2, 2), padding='valid')

    branch3x3dbl = conv2d_bn(x4, 64, 1, 1)
    branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3)
    branch3x3dbl = conv2d_bn(branch3x3dbl, 96, 3, 3, strides=(2, 2), padding='valid')

    branch_pool = layers.MaxPooling2D((3, 3), strides=(2, 2))(x4)

    # mixed 3 (elu): 17 x 17 x 768
    branch3x3_1 = conv2d_bn1(x4, 384, 3, 3, strides=(2, 2), padding='valid')

    branch3x3dbl_1 = conv2d_bn1(x4, 64, 1, 1)
    branch3x3dbl_1 = conv2d_bn1(branch3x3dbl_1, 96, 3, 3)
    branch3x3dbl_1 = conv2d_bn1(branch3x3dbl_1, 96, 3, 3, strides=(2, 2), padding='valid')

    branch_pool_1 = layers.MaxPooling2D((3, 3), strides=(2, 2))(x4)



    x5 = layers.concatenate(
        [branch3x3, branch3x3dbl, branch_pool, branch3x3_1, branch3x3dbl_1, branch_pool_1],
        axis=channel_axis,
        name='mixed3')

    # mixed 4 (relu): 17 x 17 x 768
    branch1x1 = conv2d_bn(x5, 192, 1, 1)

    branch7x7 = conv2d_bn(x5, 128, 1, 1)
    branch7x7 = conv2d_bn(branch7x7, 128, 1, 7)
    branch7x7 = conv2d_bn(branch7x7, 192, 7, 1)

    branch7x7dbl = conv2d_bn(x5, 128, 1, 1)
    branch7x7dbl = conv2d_bn(branch7x7dbl, 128, 7, 1)
    branch7x7dbl = conv2d_bn(branch7x7dbl, 128, 1, 7)
    branch7x7dbl = conv2d_bn(branch7x7dbl, 128, 7, 1)
    branch7x7dbl = conv2d_bn(branch7x7dbl, 192, 1, 7)

    branch_pool = layers.AveragePooling2D((3, 3),
                                          strides=(1, 1),
                                          padding='same')(x5)
    branch_pool = conv2d_bn(branch_pool, 192, 1, 1)


    # mixed 4 (elu): 17 x 17 x 768
    branch1x1_1 = conv2d_bn1(x5, 192, 1, 1)

    branch7x7_1 = conv2d_bn1(x5, 128, 1, 1)
    branch7x7_1 = conv2d_bn1(branch7x7_1, 128, 1, 7)
    branch7x7_1 = conv2d_bn1(branch7x7_1, 192, 7, 1)

    branch7x7dbl_1 = conv2d_bn1(x5, 128, 1, 1)
    branch7x7dbl_1 = conv2d_bn1(branch7x7dbl_1, 128, 7, 1)
    branch7x7dbl_1 = conv2d_bn1(branch7x7dbl_1, 128, 1, 7)
    branch7x7dbl_1 = conv2d_bn1(branch7x7dbl_1, 128, 7, 1)
    branch7x7dbl_1 = conv2d_bn1(branch7x7dbl_1, 192, 1, 7)

    branch_pool_1 = layers.AveragePooling2D((3, 3),
                                          strides=(1, 1),
                                          padding='same')(x5)
    branch_pool_1 = conv2d_bn1(branch_pool_1, 192, 1, 1)




    x6 = layers.concatenate([branch1x1, branch7x7, branch7x7dbl, branch_pool, branch1x1_1, branch7x7_1, branch7x7dbl_1, branch_pool_1],
        axis=channel_axis,
        name='mixed4')
    
    
    
    # mixed 5, 6 (relu): 17 x 17 x 768
    x5_6_outputs = []
    for i in range(2):
        branch1x1 = conv2d_bn(x6, 192, 1, 1)
        
        branch7x7 = conv2d_bn(x6, 160, 1, 1)
        branch7x7 = conv2d_bn(branch7x7, 160, 1, 7)
        branch7x7 = conv2d_bn(branch7x7, 192, 7, 1)
        
        branch7x7dbl = conv2d_bn(x6, 160, 1, 1)
        branch7x7dbl = conv2d_bn(branch7x7dbl, 160, 7, 1)
        branch7x7dbl = conv2d_bn(branch7x7dbl, 160, 1, 7)
        branch7x7dbl = conv2d_bn(branch7x7dbl, 160, 7, 1)
        branch7x7dbl = conv2d_bn(branch7x7dbl, 192, 1, 7)
        
        branch_pool = layers.AveragePooling2D((3, 3), strides=(1, 1), padding='same')(x6)
        branch_pool = conv2d_bn(branch_pool, 192, 1, 1)

        # Concatenate and store the result for this iteration
        x_mixed = layers.concatenate(
            [branch1x1, branch7x7, branch7x7dbl, branch_pool],
            axis=channel_axis,
            name='mixed' + str(5 + i))
        x5_6_outputs.append(x_mixed)

    # The result of mixed 5, 6 is the concatenation of the iterations' outputs
    x5_6_concat = layers.concatenate(x5_6_outputs, axis=channel_axis, name='mixed5_6_concat')

    # mixed 5, 6 (elu): 17 x 17 x 768
    x7_8_outputs = []
    for i in range(2):
        branch1x1_1 = conv2d_bn1(x6, 192, 1, 1)
        
        branch7x7_1 = conv2d_bn1(x6, 160, 1, 1)
        branch7x7_1 = conv2d_bn1(branch7x7_1, 160, 1, 7)
        branch7x7_1 = conv2d_bn1(branch7x7_1, 192, 7, 1)
        
        branch7x7dbl_1 = conv2d_bn1(x6, 160, 1, 1)
        branch7x7dbl_1 = conv2d_bn1(branch7x7dbl_1, 160, 7, 1)
        branch7x7dbl_1 = conv2d_bn1(branch7x7dbl_1, 160, 1, 7)
        branch7x7dbl_1 = conv2d_bn1(branch7x7dbl_1, 160, 7, 1)
        branch7x7dbl_1 = conv2d_bn1(branch7x7dbl_1, 192, 1, 7)
        
        branch_pool_1 = layers.AveragePooling2D((3, 3), strides=(1, 1), padding='same')(x6)
        branch_pool_1 = conv2d_bn1(branch_pool_1, 192, 1, 1)

        # Concatenate and store the result for this iteration
        x_mixed_1 = layers.concatenate(
            [branch1x1_1, branch7x7_1, branch7x7dbl_1, branch_pool_1],
            axis=channel_axis,
            name='mixed' + str(7 + i))
        x7_8_outputs.append(x_mixed_1)

    # The result of mixed 7, 8 is the concatenation of the iterations' outputs
    x7_8_concat = layers.concatenate(x7_8_outputs, axis=channel_axis, name='mixed7_8_concat')

    # Finally, concatenate the results of mixed 5, 6 and mixed 7, 8
    x9 = layers.concatenate([x5_6_concat, x7_8_concat], axis=channel_axis, name='mixed9')

    

    # mixed 10 (relu): 17 x 17 x 768
    branch1x1 = conv2d_bn(x9, 192, 1, 1)

    branch7x7 = conv2d_bn(x9, 192, 1, 1)
    branch7x7 = conv2d_bn(branch7x7, 192, 1, 7)
    branch7x7 = conv2d_bn(branch7x7, 192, 7, 1)

    branch7x7dbl = conv2d_bn(x9, 192, 1, 1)
    branch7x7dbl = conv2d_bn(branch7x7dbl, 192, 7, 1)
    branch7x7dbl = conv2d_bn(branch7x7dbl, 192, 1, 7)
    branch7x7dbl = conv2d_bn(branch7x7dbl, 192, 7, 1)
    branch7x7dbl = conv2d_bn(branch7x7dbl, 192, 1, 7)

    branch_pool = layers.AveragePooling2D((3, 3),
                                          strides=(1, 1),
                                          padding='same')(x9)
    branch_pool = conv2d_bn(branch_pool, 192, 1, 1)


    # mixed 7 (elu): 17 x 17 x 768
    branch1x1_1 = conv2d_bn1(x9, 192, 1, 1)

    branch7x7_1 = conv2d_bn1(x9, 192, 1, 1)
    branch7x7_1 = conv2d_bn1(branch7x7_1, 192, 1, 7)
    branch7x7_1 = conv2d_bn1(branch7x7_1, 192, 7, 1)

    branch7x7dbl_1 = conv2d_bn1(x9, 192, 1, 1)
    branch7x7dbl_1 = conv2d_bn1(branch7x7dbl_1, 192, 7, 1)
    branch7x7dbl_1 = conv2d_bn1(branch7x7dbl_1, 192, 1, 7)
    branch7x7dbl_1 = conv2d_bn1(branch7x7dbl_1, 192, 7, 1)
    branch7x7dbl_1 = conv2d_bn1(branch7x7dbl_1, 192, 1, 7)

    branch_pool_1 = layers.AveragePooling2D((3, 3),
                                          strides=(1, 1),
                                          padding='same')(x9)
    branch_pool_1 = conv2d_bn1(branch_pool_1, 192, 1, 1)


    x10 = layers.concatenate([branch1x1, branch7x7, branch7x7dbl, branch_pool, branch1x1_1, branch7x7_1, branch7x7dbl_1, branch_pool_1],
        axis=channel_axis,
        name='mixed10')


    # mixed 11 (relu): 8 x 8 x 1280
    branch3x3 = conv2d_bn(x10, 192, 1, 1)
    branch3x3 = conv2d_bn(branch3x3, 320, 3, 3,
                          strides=(2, 2), padding='valid')

    branch7x7x3 = conv2d_bn(x10, 192, 1, 1)
    branch7x7x3 = conv2d_bn(branch7x7x3, 192, 1, 7)
    branch7x7x3 = conv2d_bn(branch7x7x3, 192, 7, 1)
    branch7x7x3 = conv2d_bn(branch7x7x3, 192, 3, 3, strides=(2, 2), padding='valid')

    branch_pool = layers.MaxPooling2D((3, 3), strides=(2, 2))(x10)

    # mixed 11 (elu): 8 x 8 x 1280
    branch3x3_1 = conv2d_bn1(x10, 192, 1, 1)
    branch3x3_1 = conv2d_bn1(branch3x3_1, 320, 3, 3,
                          strides=(2, 2), padding='valid')

    branch7x7x3_1 = conv2d_bn1(x10, 192, 1, 1)
    branch7x7x3_1 = conv2d_bn1(branch7x7x3_1, 192, 1, 7)
    branch7x7x3_1 = conv2d_bn1(branch7x7x3_1, 192, 7, 1)
    branch7x7x3_1 = conv2d_bn1(branch7x7x3_1, 192, 3, 3, strides=(2, 2), padding='valid')

    branch_pool_1 = layers.MaxPooling2D((3, 3), strides=(2, 2))(x10)



    x11 = layers.concatenate([branch3x3, branch7x7x3, branch_pool, branch3x3_1, branch7x7x3_1, branch_pool_1],
        axis=channel_axis,
        name='mixed11')


    # mixed 12 (relu): 8 x 8 x 2048
    x12_outputs = []
    for i in range(2):
        branch1x1 = conv2d_bn(x11, 320, 1, 1)

        branch3x3 = conv2d_bn(x11, 384, 1, 1)
        branch3x3_1 = conv2d_bn(branch3x3, 384, 1, 3)
        branch3x3_2 = conv2d_bn(branch3x3, 384, 3, 1)
        branch3x3 = layers.concatenate(
            [branch3x3_1, branch3x3_2],
            axis=channel_axis,
            name='mixed12_' + str(i))

        branch3x3dbl = conv2d_bn(x11, 448, 1, 1)
        branch3x3dbl = conv2d_bn(branch3x3dbl, 384, 3, 3)
        branch3x3dbl_1 = conv2d_bn(branch3x3dbl, 384, 1, 3)
        branch3x3dbl_2 = conv2d_bn(branch3x3dbl, 384, 3, 1)
        branch3x3dbl = layers.concatenate(
            [branch3x3dbl_1, branch3x3dbl_2], axis=channel_axis)

        branch_pool = layers.AveragePooling2D(
            (3, 3), strides=(1, 1), padding='same')(x11)
        branch_pool = conv2d_bn(branch_pool, 192, 1, 1)

        x12 = layers.concatenate([branch1x1, branch3x3, branch3x3dbl, branch_pool],
                                 axis=channel_axis,
                                 name='mixed' + str(12 + i))
        x12_outputs.append(x12)

    # Assuming you want to concatenate the outputs of the two iterations:
    x12_concat = layers.concatenate(x12_outputs, axis=channel_axis, name='mixed12_concat')

    # mixed 14 (elu): 8 x 8 x 2048
    x13_outputs = []
    for i in range(2):
        branch1x1_1 = conv2d_bn1(x11, 320, 1, 1)

        branch3x3_1 = conv2d_bn1(x11, 384, 1, 1)
        branch3x3_1_1 = conv2d_bn1(branch3x3_1, 384, 1, 3)
        branch3x3_2_1 = conv2d_bn1(branch3x3_1, 384, 3, 1)
        branch3x3_1 = layers.concatenate([branch3x3_1_1, branch3x3_2_1],
                                         axis=channel_axis,
                                         name='mixed14_' + str(i))

        branch3x3dbl_1 = conv2d_bn1(x11, 448, 1, 1)
        branch3x3dbl_1 = conv2d_bn1(branch3x3dbl_1, 384, 3, 3)
        branch3x3dbl_1_1 = conv2d_bn1(branch3x3dbl_1, 384, 1, 3)
        branch3x3dbl_2_1 = conv2d_bn1(branch3x3dbl_1, 384, 3, 1)
        branch3x3dbl_1 = layers.concatenate(
            [branch3x3dbl_1_1, branch3x3dbl_2_1], axis=channel_axis)

        branch_pool_1 = layers.AveragePooling2D((3, 3), strides=(1, 1), padding='same')(x11)
        branch_pool_1 = conv2d_bn1(branch_pool_1, 192, 1, 1)

        x13 = layers.concatenate([branch1x1_1, branch3x3_1, branch3x3dbl_1, branch_pool_1],
                                 axis=channel_axis,
                                 name='mixed' + str(14 + i))
        x13_outputs.append(x13)

    # Assuming you want to concatenate the outputs of the two iterations:
    x13_concat = layers.concatenate(x13_outputs, axis=channel_axis, name='mixed14_concat')

    # Finally, concatenate the results of mixed12 and mixed14
    x14 = layers.concatenate([x12_concat, x13_concat], axis=channel_axis, name='mixed16')
    x14 = layers.BatchNormalization()(x14)
    x14 = layers.GlobalAveragePooling2D()(x14)
    x14 = layers.Dropout(0.7)(x14)
    x14 = layers.Dense(1024, activation='relu')(x14)
    x14 = layers.BatchNormalization()(x14)
    x14 = layers.Dropout(0.5)(x14)

    outputs = layers.Dense(num_classes, activation='softmax')(x14)

    model = keras.Model(inputs=input_img, outputs=outputs)

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'], run_eagerly=True )
    return model


# Instantiate the Deep CNN model
image_height, image_width, num_channels = 312, 312, 3
num_classes = 35  # Replace this with the actual number of classes, e.g., len(set(labels))
model = build_custom_inception_model(input_shape=(image_height, image_width, num_channels), num_classes=num_classes)

model.summary()


#MODEL TRAINING
#ReduceLROnPlateau
from tensorflow.keras.callbacks import LearningRateScheduler, TensorBoard, CSVLogger, EarlyStopping, ModelCheckpoint,  ReduceLROnPlateau
from datetime import datetime

# Define a CSV logger to save training history to a CSV file
csv_logger = CSVLogger('pmodifiedinceptionHE_2000SP35_1024h4.csv')


reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',  # Monitor validation accuracy
    factor=0.2,             # Factor by which the learning rate will be reduced
    patience=5,             # Number of epochs with no improvement after which learning rate will be reduced
    verbose=1,               # If 1, prints messages to stdout when reducing learning rate
    min_lr=0.00001           # Lower bound on the learning rate
)


# Callbacks and training
early_stopping = EarlyStopping(patience=10, restore_best_weights=True)
model_checkpoint = ModelCheckpoint('pmodifiedinceptionHE_2000SP35_1024h4.h5', save_best_only=True)
callbacks = [csv_logger, early_stopping, reduce_lr, model_checkpoint]

# Data augmentation setup
datagen = ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest')
datagen.fit(X_train)

begin = datetime.now()

# Training with data augmentation
history = model.fit(
    datagen.flow(X_train, y_train_encoded, batch_size=32),
    validation_data=(X_val, y_val_encoded),
    epochs=100,
    callbacks=callbacks
)

finish = datetime.now()

# Evaluate the model on the test set
test_loss, test_accuracy = model.evaluate(X_test, y_test_encoded)
print(f"Test accuracy: {test_accuracy * 100:.2f}%")


print(f"Training started at: {begin}")
print(f"Training finished at: {finish}")

total_time = finish-begin
print('Total Training Time (In Seconds) : ', total_time.total_seconds())

# Save the final model in .keras format
model.save('pmodifiedinceptionHE_2000SP35_1024h4.keras')
model.save('pmodifiedinceptionHE_2000SP35_1024h4.h5')
