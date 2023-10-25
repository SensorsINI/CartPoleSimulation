import tensorflow as tf
from tensorflow import keras

class Plastic_Dense_Layer(keras.layers.Layer):
    def __init__(self, input_dim, unit, activation=True):
        super(Plastic_Dense_Layer, self).__init__()
        self.w = self.add_weight(
            name='w', shape=(input_dim, unit), initializer="glorot_uniform", trainable=True
        )
        self.b = self.add_weight(
            name='b', shape=(unit, ), initializer="zeros", trainable=True
        )
        self.alpha = self.add_weight(
            name='alpha', shape=(input_dim, unit), initializer="glorot_normal", trainable=True
        )
        self.eta = tf.Variable(
            name='eta', initial_value=0.00001, trainable=True
        )
        self.hebb = tf.Variable(
            name='hebb', initial_value=tf.zeros([input_dim, unit]), trainable=False
        )
        self.updateterm = tf.Variable(
            name='updateterm', initial_value=tf.zeros([input_dim, unit]), trainable=False
        )
        self.activation = activation
    
    def call(self, input, training=None):
        output = tf.matmul(input, (tf.multiply(self.alpha, ((1-self.eta)*self.hebb+self.eta*self.updateterm))+self.w))+self.b
        if self.activation:
            output = tf.tanh(output)

        if training:
            new = (1-self.eta)*self.hebb+self.eta*self.updateterm
            self.hebb.assign(new)

        if training:
            product = tf.matmul(tf.expand_dims(input,3), tf.expand_dims(output,2))
            bs = product.shape[0]
            ts = product.shape[1]
            update = tf.reduce_sum(product, [0,1])
            if bs is not None:
                update = update / int(bs)
            if ts is not None:
                update = update / int(ts)
            self.updateterm.assign(update)

            # Oja's rule
            # product1 = tf.matmul(tf.expand_dims(input,3), tf.expand_dims(output,2))
            # bs = product1.shape[0]
            # ts = product1.shape[1]
            # product20 = tf.multiply(tf.expand_dims(output,2) ,self.hebb)
            # product21 = tf.multiply(tf.expand_dims(output,2), product20)
            # update = product1 - product21
            # update = tf.reduce_sum(update, [0,1])
            # if bs is not None:
            #     update = update / int(bs)
            # if ts is not None:
            #     update = update / int(ts)
            # new = self.hebb + self.eta*update
            # self.hebb.assign(new)

        return output

#class Plastic_Dense_Net(keras.Model):