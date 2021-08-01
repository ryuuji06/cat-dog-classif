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
from tensorflow.keras.models import load_model
#from tensorflow.keras.layers.experimental.preprocessing import RandomFlip, RandomRotation, RandomTranslation, Rescaling
#from tensorflow.keras.layers import Conv2D, SeparableConv2D, MaxPooling2D, GlobalAveragePooling2D
#from tensorflow.keras.layers import Activation, BatchNormalization, Dropout, Dense, add
#from tensorflow.keras.callbacks import ModelCheckpoint
#from tensorflow.keras.optimizers import Adam


# execute file from python interpreter
# exec(open('t00_dataset_object.py').read())

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

# ---------------------------------------------------------------------


testfolder = "test01"
model_number = "5"
dataset_path = "C:/datasets/PetImages"
RNG_SEED = 33


# =====================================================
print('\n(1) Load History and Model')
# =====================================================

image_size = (180, 180)
batch_size = 32

# # (1) load training history
# # failed to save history
# with open(testfolder+"/hist.pickle", 'rb') as f:
#     history = pickle.load(f)

# (2) load model
model = load_model("%s/model_at_%s.h5"%(testfolder,model_number))

test_ds = image_dataset_from_directory(
    dataset_path,
    seed=RNG_SEED,
    image_size=image_size,
    batch_size=batch_size,
)

# cat=0, dog=1
for x,y in test_ds.take(1):
	for i in range(2):
		print(x.shape)
		print(y)

# plt.figure(figsize=(10, 10))
# for images, labels in test_ds.take(1):
# 	for i in range(9):
# 		ax = plt.subplot(3, 3, i + 1)
# 		plt.imshow(images[i].numpy().astype("uint8"))
# 		plt.title(int(labels[i]))
# 		plt.axis("off")
# plt.show()


# =====================================================
print('\n(2) Verify Performance')
# =====================================================

# (1) plot history (loss and accuracy)

# (2) verify model layer weghits and shapes
#   (all kernels are small 3x3)
#	Conv2D: kernel (3x3x3x32) (kernel size 3x3 for each input, for each output map) and bias (32)
#   BatchNorm: gamma, beta, moving_mean and moving_variance (all of length 32)


# (3) Plot some of the learned kernels



# =====================================================
print('\n(3) Prediction')
# =====================================================

# # load individual images
# img = keras.preprocessing.image.load_img("PetImages/Cat/6779.jpg", target_size=image_size)
# img_array = keras.preprocessing.image.img_to_array(img)
# img_array = tf.expand_dims(img_array, 0)  # Create batch axis

predictions = model.predict(x)

print("Prob.\tPred.\tTrue\n")
for i in range(batch_size):
	print("%.2f\t%d\t%d"%(predictions[i],np.int32(np.round(predictions[i])),y[i]))




# ==================================================================