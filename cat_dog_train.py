# clear existing user defined variables
for element in dir():
    if element[0:2] != "__":
        del globals()[element]

import os
import pickle
from time import time
from tensorflow.compat import as_bytes
from matplotlib import pyplot as plt

import numpy as np
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras import Sequential, Model, Input
from tensorflow.keras.layers.experimental.preprocessing import RandomFlip, RandomRotation, RandomTranslation, Rescaling
from tensorflow.keras.layers import Conv2D, SeparableConv2D, MaxPooling2D, GlobalAveragePooling2D
from tensorflow.keras.layers import Activation, BatchNormalization, Dropout, Dense, add
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.optimizers import Adam



# execute file from python interpreter
# exec(open('cat_dog_train.py').read())

# =================================================================
# =================================================================
# SIMPLE IMAGE CLASSIFICATION TASK (BINARY - CAT OR DOG)
# =================================================================
# =================================================================
# https://keras.io/examples/vision/image_classification_from_scratch/

# PetImages
# - originally, the dataset has
#   12501 images of cats
#   12501 images of dogs

# model has ~2 million parameters
# each epoch takes ~4000s (1h15min)
# still found some corrupted images during training

# about separable convolutions
# https://towardsdatascience.com/a-basic-introduction-to-separable-convolutions-b99ec3102728

# ---------------------------------------------------------------------

test = '02'
dataset_path = "C:/datasets/PetImages"
RNG_SEED = 1333

epochs = 5

resultfolder = 'test%s'%(test)
if os.path.exists(resultfolder):
	raise AssertionError('Folder already exists. Be sure to use a proper name for the test.')
else:
    os.makedirs(resultfolder)


# =====================================================
print('\n(0) Remove Corrupted Images') # only in the first run
# =====================================================
# took ~340s

# t1 = time()
# num_skipped = 0
# for subfolder in ("Cat", "Dog"):
# 	subfolder_path = os.path.join(dataset_path, subfolder)
# 	for fname in os.listdir(subfolder_path):
# 		fpath = os.path.join(subfolder_path, fname)
# 		try:
# 			fobj = open(fpath, "rb")
# 			is_jfif = as_bytes("JFIF") in fobj.peek(10)
# 		finally:
# 			fobj.close()
# 		if not is_jfif:
# 			num_skipped += 1
# 			# Delete corrupted image
# 			os.remove(fpath)

# print("Deleted %d images" % num_skipped)
# t2 = time(); print('Elapsed time: %.3fs'%(t2-t1))


# =====================================================
print('\n(1) Load Data and Generate Dataset')
# =====================================================

image_size = (180, 180)
batch_size = 32

train_ds = image_dataset_from_directory(
    dataset_path,
    validation_split=0.2,
    subset="training",
    seed=RNG_SEED,
    image_size=image_size,
    batch_size=batch_size,
)

val_ds = image_dataset_from_directory(
    dataset_path,
    validation_split=0.2,
    subset="validation",
    seed=RNG_SEED,
    image_size=image_size,
    batch_size=batch_size,
)

# buffered prefetching so we can yield data from disk without having I/O becoming blocking
train_ds = train_ds.prefetch(buffer_size=32)
val_ds = val_ds.prefetch(buffer_size=32)

plt.figure(figsize=(10, 10))
for images, labels in train_ds.take(1):
	for i in range(9):
		ax = plt.subplot(3, 3, i + 1)
		plt.imshow(images[i].numpy().astype("uint8"))
		plt.title(int(labels[i]))
		plt.axis("off")
plt.show()


# =====================================================
print('\n(2) Build Model')
# =====================================================

# DATA AUGMENTATION: flip, rotation and translation
# PREPROCESSING: rescale interval [0,255] to [0,1]

# default fill_mode of RandomRotation/RandomTranslation is 'reflect'
data_augmentation = Sequential([
	RandomFlip("horizontal"),
	RandomRotation(0.1), # range of +-0.1 of 2pi
	RandomTranslation(0.1,0.1)
])

def make_model(input_shape):

	inputs = Input(shape=input_shape)

	# Image augmentation block
	x = data_augmentation(inputs)
	x = Rescaling(1.0 / 255)(x)

	# Entry block
	x = Conv2D(32, 3, strides=2, padding="same")(x)
	x = BatchNormalization()(x)
	x = Activation("relu")(x)

	x = Conv2D(64, 3, padding="same")(x)
	x = BatchNormalization()(x)
	x = Activation("relu")(x)

	previous_block_activation = x  # Set aside residual

	for size in [128, 256, 512, 728]:		
		x = Activation("relu")(x)
		x = SeparableConv2D(size, 3, padding="same")(x)
		x = BatchNormalization()(x)

		x = Activation("relu")(x)
		x = SeparableConv2D(size, 3, padding="same")(x)
		x = BatchNormalization()(x)

		x = MaxPooling2D(3, strides=2, padding="same")(x)

		# Project residual
		residual = Conv2D(size, 1, strides=2, padding="same")(previous_block_activation)
		x = add([x, residual])  # Add back residual
		previous_block_activation = x  # Set aside next residual

	x = SeparableConv2D(1024, 3, padding="same")(x)
	x = BatchNormalization()(x)
	x = Activation("relu")(x)

	x = GlobalAveragePooling2D()(x)

	x = Dropout(0.5)(x)
	outputs = Dense(1, activation="sigmoid")(x)
	return Model(inputs, outputs)


model = make_model(input_shape=image_size + (3,) )
model.summary()



# =====================================================
print('\n(3) Training')
# =====================================================

callbacks = [ ModelCheckpoint(resultfolder+"/model-{epoch}-{val_acc:.3f}.h5", save_best_only=True) ]
model.compile(
    optimizer=Adam(1e-3),
    loss="binary_crossentropy",
    metrics=["accuracy"],
)

h = model.fit(train_ds, validation_data=val_ds, epochs=epochs, callbacks=callbacks)

hist = [h.history['loss'], h.history['val_loss'],h.history['acc'],h.history['val_acc']]

# save model loss history
with open(resultfolder+"/hist.pickle", 'wb') as f:
    pickle.dump(hist.history, f)

#plt.figure(1)
#plt.plot(hist)





# ==================================================================