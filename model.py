#-*- coding: utf-8 -*-
# @Time     :  上午11:40
# @Author   : ergouzi
# @FileName : model.py

import tensorflow as tf

class Model(object):
    
    def __init__(self):
        pass

    def conv(self, input_data, shape, step_shape, out_channel, name, has_relu = True):
        """
        使用shape为卷积核的两个维度来进行卷积
        对于shape表示了卷积核的维度 [1, 2, 3, 4]  分别表示了 height, width, in_channel, out_channel
        step_shape表示了strides 分别表示了 ... 一般取 [1, 1, 1, 1]
        """
        input_channel = input_data.get_shape()[-1]
        shape.append(input_channel)
        shape.append(out_channel)

        with tf.variable_scope(name):
            weight = tf.get_variable(name="weights", shape=shape, dtype=tf.float32, initializer=tf.glorot_uniform_initializer())
            bias = tf.get_variable(name="bias", shape=[out_channel], dtype=tf.float32, initializer=tf.glorot_uniform_initializer())
        conv = tf.nn.conv2d(input_data, weight, step_shape, "SAME")
        res = tf.nn.bias_add(conv, bias=bias)
        if has_relu:
            relu_res = tf.nn.relu(res)
        else:
            relu_res = res
        return relu_res

    def fc(self, input_data, out_channel, name):
        """
        进行全连接，第一个维度为batch_size , 最后输出为指定的output_channel维度

        """
        size_in = 1   # 计算最后需要进行进行全连接操作的维度
        for i, size_axis_i in enumerate(input_data.get_shape()):
            if i == 0:
                continue
            else:
                # 注意这里获得的不是value，而是Dimension对象，调用Dimension的value可以获得int类型的数据
                size_in *= size_axis_i.value
        input_data_flat = tf.reshape(input_data, shape=[-1, size_in])
        # 这里一定要使用variable_scope来设定不同的scope
        with tf.variable_scope(name):
            weight = tf.get_variable(name="weights", shape=[size_in, out_channel], dtype=tf.float32, initializer=tf.glorot_uniform_initializer())
            bias = tf.get_variable(name= "bias", dtype=tf.float32, shape=[out_channel], initializer=tf.glorot_uniform_initializer())
        res = tf.matmul(input_data, weight)
        out = tf.nn.relu(tf.nn.bias_add(res, bias))
        return out

    def max_pool(self, input_data, shape, step_shape, name):
        """最大池化
        shape 各个维度分别表示了在 batch_size, height, width, channel 中的池化
        step ... 对应的维度 bs h w c
        """
        out = tf.nn.max_pool(input_data, shape, step_shape, padding="SAME", name=name)
        return out

    def flatten(self, input_data, name):
        """ 对于输入的数据，将其转化为 batch_size, all_feature的维度"""
        size = 1
        for i, s in enumerate(input_data.get_shape()):
            if i == 0:
                continue
            else:
                size *= s.value
        with tf.name_scope(name=name):
            return tf.reshape(input_data, shape=[-1, size], name="reshape")

    def concat(self, input_1, input_2, name):
        """对于指定的维度进行concat操作"""
        # 对于最后的经过了flatten的tensor，直接对axis=1维度进行连接
        concat = tf.concat([input_1, input_2], axis=1, name=name)
        return concat