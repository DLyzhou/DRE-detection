import tensorflow as tf
from tensorflow.python import pywrap_tensorflow


model_reader = tf.train.NewCheckpointReader(r"./Model/model.ckpt")

var_dict = model_reader.get_variable_to_shape_map()

for key in var_dict:
    print("variable name: ", key)
    print(model_reader.get_tensor(key))
