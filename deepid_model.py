#-*- coding: utf-8 -*-
# @Time     :  上午11:32
# @Author   : ergouzi
# @FileName : deepid_model.py
from model import *


class DeepId(Model):
    
    def __init__(self):
        self.num_classes = 5749
        # placeholder
        self.input_placeholder = tf.placeholder(tf.float32, shape=[None, 39, 31, 3], name="input_img")
        # 分类为 5750个人

        self.label_placeholder = tf.placeholder(tf.int32, shape=[None, self.num_classes])

        self.drop_keep_prob_placeholder = tf.placeholder(tf.float32, name="keep_prob")

        self.add_all_layers()
        self.add_loss_acc()

    def add_all_layers(self):
        """具体的网络的结构"""
        self.conv1 = self.conv(input_data=self.input_placeholder, shape=[4, 4], step_shape=[1, 1, 1, 1], out_channel=20, name="conv1")
        self.pool1 = self.max_pool(input_data=self.conv1, shape=[1, 2, 2, 1], step_shape=[1, 2, 2, 1], name="pool1")
        self.conv2 = self.conv(input_data=self.pool1, shape=[3, 3], step_shape=[1, 1, 1, 1], out_channel=40, name="conv2")
        self.pool2 = self.max_pool(input_data=self.conv2, shape=[1, 2, 2, 1], step_shape=[1, 2, 2, 1], name="pool2")
        self.conv3 = self.conv(input_data=self.pool2, shape=[3, 3], step_shape=[1, 1, 1, 1], out_channel=60, name="conv3", has_relu=False)
        self.pool3 = self.max_pool(input_data=self.conv3, shape=[1, 2, 2, 1], step_shape=[1, 2, 2, 1], name="pool3")
        # 卷积到这一步的时候，分为两个分支

        # 分支 1 先卷积，再flatten
        self.conv4 = self.conv(input_data=self.pool3, shape=[2, 2], step_shape=[1, 1, 1, 1], out_channel=80, name="conv4", has_relu=False)
        self.pool4 = self.max_pool(input_data=self.conv4, shape=[1, 2, 2, 1], step_shape=[1, 1, 1, 1], name="pool4")
        self.flatten1 = self.flatten(self.pool4, name="flatten1")

        # 分支 2 直接flatten
        self.flatten2 = self.flatten(self.pool3, name="flatten2")

        # 直接对flatten1 ,2 进行连接操作
        self.concat = self.concat(self.flatten1, self.flatten2, "concat")

        # 进行全连接操作
        self.fc1 = self.fc(self.concat, 160, "embedding_face")
        self.embedding = self.fc1

        self.relu_embedding = tf.nn.relu(self.embedding, name="relu_embdding")
        # 对 fc1进行dropout
        self.h_embedding_drop = tf.nn.dropout(self.relu_embedding, keep_prob=self.drop_keep_prob_placeholder, name="embedding_dropout")
        # 进行全连接到 5750维
        self.fc2 = self.fc(self.h_embedding_drop, self.num_classes, "class_belong")

    def add_loss_acc(self):
        """设置损失函数和精确度"""
        # 损失函数
        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.label_placeholder, logits=self.fc2))
        # 准确率 这里准确率的设置 tf.equal获得到的tensor类型为bool，这里需要通过tf.cast将其转化为float类型
        correct_prediction = tf.equal(tf.argmax(self.fc2, 1), tf.argmax(self.label_placeholder, 1))
        self.acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
