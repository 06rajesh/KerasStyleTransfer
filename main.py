#!/usr/bin/python3

from __future__ import print_function

import time
from PIL import Image
import numpy as np

from keras import backend
from keras.applications.vgg16 import VGG16

from scipy.optimize import fmin_l_bfgs_b
from scipy.misc import imsave

from style_transfer import Utils, Evaluator

height = 512
width = 512

content_image_path = "images/hugo.png"
content_image = Image.open(content_image_path)
content_image = content_image.resize((width, height))

style_image_path = 'images/wave.png'
style_image = Image.open(style_image_path)
style_image = style_image.resize((width, height))

utils = Utils(width, height, backend)

content_array = utils.image_preprocess(content_image)
style_array = utils.image_preprocess(style_image)

content_image = backend.variable(content_array)
style_image = backend.variable(style_array)
combination_image = backend.placeholder((1, height, width, 3))

input_tensor = backend.concatenate([content_image, style_image, combination_image], axis=0)

model = VGG16(input_tensor=input_tensor, weights='imagenet', include_top=False)
layers = dict([(layer.name, layer.output) for layer in model.layers])

content_weight = 0.025
style_weight = 5.0
total_variation_weight = 1.0

loss = backend.variable(0.0)


layer_features = layers['block2_conv2']
content_image_features = layer_features[0, :, :, :]
combination_features = layer_features[2, :, :, :]

loss += content_weight * utils.content_loss(content_image_features, combination_features)

feature_layers = ['block1_conv2', 'block2_conv2',
                  'block3_conv3', 'block4_conv3',
                  'block5_conv3']

for layer_name in feature_layers:
    layer_features = layers[layer_name]
    style_features = layer_features[1, :, :, :]
    combination_features = layer_features[2, :, :, :]
    sl = utils.style_loss(style_features, combination_features)
    loss += (style_weight / len(feature_layers)) * sl

loss += total_variation_weight * utils.total_variation_loss(combination_image)
grads = backend.gradients(loss, combination_image)

outputs = [loss]
outputs += grads
f_outputs = backend.function([combination_image], outputs)


evaluator = Evaluator(width, height, f_outputs)

x = np.random.uniform(0, 255, (1, height, width, 3)) - 128

iteration = 1

for i in range(iteration):
    print('Start of iteration', i)
    start_time = time.time()
    x, min_val, info = fmin_l_bfgs_b(evaluator.loss, x.flatten(), fprime=evaluator.grads, maxfun=20)
    print('Current loss value:', min_val)
    end_time = time.time()
    print('Iteration %d completed in %ds' % (i, end_time - start_time))


x = x.reshape((height, width, 3))
x = x[:, :, ::-1]
x[:, :, 0] += 103.939
x[:, :, 1] += 116.779
x[:, :, 2] += 123.68
x = np.clip(x, 0, 255).astype('uint8')

result = Image.fromarray(x)
imsave('images/output.png', result)
