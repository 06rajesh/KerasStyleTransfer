#!/usr/bin/python3
import numpy as np


class Utils(object):

    def __init__(self, width, height, backend):
        self.width = width
        self.height = height
        self.k = backend

    @staticmethod
    def image_preprocess(image):
        image_array = np.asarray(image, dtype='float32')
        image_array = np.expand_dims(image_array, axis=0)

        image_array[:, :, :, 0] -= 103.939
        image_array[:, :, :, 1] -= 116.779
        image_array[:, :, :, 2] -= 123.68
        image_array = image_array[:, :, :, ::-1]
        return image_array

    def image_postprocess(self, x):
        x = x.reshape((self.height, self.width, 3))
        x = x[:, :, ::-1]
        x[:, :, 0] += 103.939
        x[:, :, 1] += 116.779
        x[:, :, 2] += 123.68
        x = np.clip(x, 0, 255).astype('uint8')
        return x

    def content_loss(self, content, combination):
        return self.k.sum(self.k.square(combination - content))

    def gram_matrix(self, x):
        features = self.k.batch_flatten(self.k.permute_dimensions(x, (2, 0, 1)))
        gram = self.k.dot(features, self.k.transpose(features))
        return gram

    def style_loss(self, style, combination):
        s = self.gram_matrix(style)
        c = self.gram_matrix(combination)
        channels = 3
        size = self.width * self.height
        return self.k.sum(self.k.square(s - c)) / (4.0 * (channels ** 2) * (size ** 2))

    def total_variation_loss(self, x):
        a = self.k.square(x[:, :self.height - 1, :self.width - 1, :] - x[:, 1:, :self.width - 1, :])
        b = self.k.square(x[:, :self.height - 1, :self.width - 1, :] - x[:, :self.height - 1, 1:, :])
        return self.k.sum(self.k.pow(a + b, 1.25))


class Evaluator(object):

    def __init__(self, width, height, f_outputs):
        self.width = width
        self.height = height
        self.f_outputs = f_outputs
        self.loss_value = None
        self.grads_values = None

    def eval_loss_and_grads(self, x):
        x = x.reshape((1, self.height, self.width, 3))
        outs = self.f_outputs([x])
        loss_value = outs[0]
        grad_value = outs[1].flatten().astype('float64')
        return loss_value, grad_value

    def loss(self, x):
        assert self.loss_value is None
        loss_value, grad_values = self.eval_loss_and_grads(x)
        self.loss_value = loss_value
        self.grad_values = grad_values
        return self.loss_value

    def grads(self, x):
        assert self.loss_value is not None
        grad_values = np.copy(self.grad_values)
        self.loss_value = None
        self.grad_values = None
        return grad_values

