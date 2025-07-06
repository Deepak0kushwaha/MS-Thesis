# main architecture

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
