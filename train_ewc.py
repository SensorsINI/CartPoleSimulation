import tensorflow as tf
from tensorflow import keras
from keras.engine import data_adapter
import numpy as np
from tqdm import tqdm

def train_ewc(model, data, loss_fn, optimizer):
    weights = model.trainable_variables
    ewc_dic = {}
    for p,ele in enumerate(weights):
        ewc_dic[p] = np.zeros_like(ele.numpy())

    for i in range(1):
        data.shuffle_dataset()

        nbatch = data.calculate_number_of_batches(data.number_of_samples, data.batch_size)
        for j in tqdm([p for p in range(nbatch)]):
            x_batch,y_batch = data.get_batch(j)

            with tf.GradientTape() as tape:
                y_pred = model(x_batch, training=False)
                loss = loss_fn[0](y_batch, y_pred)
            
            gradients = tape.gradient(loss,model.trainable_variables)
        
            for q,ele in enumerate(gradients):
                ewc_dic[q] += data.batch_size*np.power(ele.numpy(),2)

            #optimizer.apply_gradients(zip(gradients,model.trainable_weights))
        
        print('loss for epoch {}: {}'.format(i, sum(loss)/len(loss)))

    for key in ewc_dic.keys():
        ewc_dic[key] /= (data.batch_size*(nbatch))
    np.save('F.npy',ewc_dic)

def get_FIM():
    F = np.load('F.npy',allow_pickle=True).item()
    return F

def get_bases():
    B = np.load('base.npy',allow_pickle=True).item()
    return B

@tf.keras.utils.register_keras_serializable(package='Custom', name='for_ewc')
class EWCRegularizer(tf.keras.regularizers.Regularizer):
  def __init__(self, l2, base):
    self.l2 = l2
    self.base = base

  def __call__(self, x):
    return 0.5*1e7*tf.math.reduce_sum(self.l2*tf.math.square(x-self.base)) # 5e7 to deny EWC;

# class EWCModel(keras.Sequential):
#     def ewc_init(self):
#         weights = self.trainable_variables
#         self.ewc_dic = {}
#         for ele in weights:
#             self.ewc_dic[ele.name] = np.zeros_like(ele.numpy())
#         self.count = 1
    
#     def train_step(self, data):
#         x, y, sample_weight = data_adapter.unpack_x_y_sample_weight(data)

#         with tf.GradientTape() as tape:
#             y_pred = self(x, training=True)
#             loss = self.compute_loss(x, y, y_pred, sample_weight)
#         self._validate_target_and_loss(y, loss)
        
#         gradients = tape.gradient(loss,self.trainable_variables)
#         print('grad')
#         print(gradients)
#         print('ewcinit')
#         print(self.ewc_dic)
#         for ele in gradients:
#             self.ewc_dic[ele.name] += np.power(ele.numpy(),2)
#         self.count += 1

#         self.optimizer.minimize(loss, self.trainable_variables, tape=tape)
#         return self.compute_metrics(x, y, y_pred, sample_weight)
    
#     def ewc_end(self):
#         for key in self.ewc_dic.keys():
#             self.ewc_dic[key] /= self.count