import os
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import utils
from imageio import imsave
import tensorflow as tf


# #Gets files and crops image
files = [os.path.join('img_align_celeba', file_i) for file_i in os.listdir('img_align_celeba') if file_i.endswith('.jpg') ]
imgs = []
for file_i in files:
    img = plt.imread(file_i)
    square = utils.imcrop_tosquare(img)
    rsz = np.array(Image.fromarray(square).resize((100,100), resample = Image.NEAREST))
    imgs.append(rsz)

imgs = np.array(imgs).astype(np.float32)
plt.figure(figsize=(10,10))
plt.imshow(utils.montage(imgs,saveto='dataset.png'))

#calculates mean of images and saves into picture
mean_img = np.mean(imgs, axis=0).astype(np.float32)
assert(mean_img.shape == (100, 100, 3))
plt.figure(figsize=(10, 10))
plt.imshow(mean_img)
plt.imsave(arr=mean_img, fname='mean.png')

#calculates standard deviation of images and saves into picture
std_img = np.std(imgs, axis=0).astype(np.float32)
assert(std_img.shape == (100, 100) or std_img.shape == (100, 100, 3))
plt.figure(figsize=(10, 10))
std_img_show = std_img / np.max(std_img)
plt.imshow(std_img_show)
plt.imsave(arr=std_img_show, fname='std.png')

#calculates normalization of images and saves into picture
norm_imgs = []
for norm_i in imgs:
    norm_imgs.append((norm_i-mean_img)/std_img)
norm_imgs = np.array(norm_imgs).astype(np.float32)
norm_imgs_show = (norm_imgs - np.min(norm_imgs)) / (np.max(norm_imgs) - np.min(norm_imgs))
plt.figure(figsize=(10, 10))
plt.imshow(utils.montage(norm_imgs_show, 'normalized.png'))

#creates kernel
ksize = 16
kernel = np.concatenate([utils.gabor(ksize)[:, :, np.newaxis] for i in range(3)], axis=2)
kernel_4d = tf.reshape(kernel, [kernel.shape[0], kernel.shape[1], 3, 1])
plt.figure(figsize=(5, 5))
plt.imshow(kernel_4d[:, :, 0, 0], cmap='gray')
plt.imsave(arr=kernel_4d[:, :, 0, 0], fname='kernel.png', cmap='gray')

#convolutes image
convolved = utils.convolve(imgs,kernel_4d)
convolved_show = (convolved - np.min(convolved)) / (np.max(convolved) - np.min(convolved))
print(convolved_show.shape)
plt.figure(figsize=(10, 10))
plt.imshow(utils.montage(convolved_show[..., 0], 'convolved.png'), cmap='gray')

flattened = tf.reshape(convolved,(100,10000))
# Now calculate some statistics about each of our images
values = tf.reduce_sum(flattened, axis=1)

# Then create another operation which sorts those values
# and then calculate the result:
idxs = tf.nn.top_k(values, k=100)[1]

# Then finally use the sorted indices to sort your images:
sorted_imgs = np.array([imgs[idx_i] for idx_i in idxs])
# Then plot the resulting sorted dataset montage:
# Make sure we have a 100 x 100 x 100 x 3 dimension array
plt.figure(figsize=(10, 10))
plt.imshow(utils.montage(sorted_imgs, 'sorted.png'))
