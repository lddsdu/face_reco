#-*- coding: utf-8 -*-
# @Time     :  下午1:17
# @Author   : ergouzi
# @FileName : model_train.py

# ① import
import tensorflow as tf
import numpy as np

from deepid_model import DeepId
from data_helper import *

# ② 参数
tf.app.flags.DEFINE_float("learning_rate", 0.001, "学习率")
tf.app.flags.DEFINE_integer("batch_size", 1024, "batch size")
tf.app.flags.DEFINE_integer("dev_interval", 1000, "do a test after train this times")
tf.app.flags.DEFINE_integer("save_interval", 20000, "save the weight after train this times")
tf.app.flags.DEFINE_float("drop_keep_prob", 0.5, "drop out")
FLAGS=tf.app.flags.FLAGS

# ③ 数据准备 在dataset中已经区分了训练数据集和测试数据集
dataset = Dataset(0.9, FLAGS.batch_size)

# ④ 网络构建
deepid = DeepId()

# ⑤ 进行网络的训练 网络的placeholder和输出
input_placeholder = deepid.input_placeholder
label_placeholder = deepid.label_placeholder
loss = deepid.loss
acc = deepid.acc

global_step = tf.Variable(0, trainable=False, name="global_step")
# 训练的过程中 global_step的值会自增１
train_op = tf.train.MomentumOptimizer(learning_rate=FLAGS.learning_rate, momentum=0.9)\
    .minimize(loss, global_step=global_step)

# 允许显存动态的增长
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

# summary
with tf.name_scope("train"):
    loss_summary = tf.summary.scalar("loss_summary", tf.reduce_mean(loss))
    acc_summary = tf.summary.scalar("acc_summary", tf.reduce_mean(acc))
train_summary_op = tf.summary.merge([loss_summary, acc_summary], "train")
train_summary_dir = os.path.join("summaries", "train")
train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

with tf.name_scope("test"):
    loss_summary_test = tf.summary.scalar("loss_summary", tf.reduce_mean(loss))
    acc_summary_test = tf.summary.scalar("acc_summary", tf.reduce_mean(acc))
test_summary_op = tf.summary.merge([loss_summary, acc_summary], "test")
test_summary_dir = os.path.join("summaries", "test")
test_summary_writer = tf.summary.FileWriter(test_summary_dir)

tf.nn.lrn()

# checkpoint 权重的存储
checkpoint_dir = "checkpoint"
checkpoint_prefix = checkpoint_dir+"/deepid-model"
if not os.path.exists(checkpoint_dir):
    os.mkdir(checkpoint_dir)

# saver
saver = tf.train.Saver()

# 这里千万不要忘记了初始化参数
sess.run(tf.global_variables_initializer())


# 定义训练和测试过程
def train_step(data, label):
    """训练一步"""
    feed_dict ={
        deepid.input_placeholder: data,
        deepid.label_placeholder: label,
        deepid.drop_keep_prob_placeholder: FLAGS.drop_keep_prob
    }
    # 怎么这么糊涂，这里没有加feed_dict。半天才发现
    loss_scalar, step, loss_acc = sess.run([train_op, global_step, train_summary_op], feed_dict=feed_dict)
    if step % 100 == 0:
        train_summary_writer.add_summary(loss_acc, global_step=step)


def val_step(data, label):
    """测试当前的准确率"""
    feed_dict ={
        deepid.input_placeholder: data,
        deepid.label_placeholder: label,
        deepid.drop_keep_prob_placeholder: 1.0              # 可以直接添加一个实数
    }
    loss_acc, step = sess.run([test_summary_op, global_step], feed_dict=feed_dict)
    if step % 100 == 0:
        test_summary_writer.add_summary(loss_acc, global_step=step)


for i in range(400000):
    label, data = dataset.next_train_batch()
    train_step(data, label)
    current_step = tf.train.global_step(sess, global_step)

    if current_step % FLAGS.dev_interval == 0:
        print "evaluation:"
        hasnext, label, step = dataset.next_test_batch()
        if hasnext:
            print "hasnext, start evaluation"
            val_step(data, label)
        else:
            print "restart, get the first batch"
            hasnext, label, step = dataset.next_test_batch()
            val_step(data, label)

    if current_step % FLAGS.save_interval == 0:
        print 'save:'
        step = tf.train.global_step(sess, global_step)
        saver.save(sess=sess, save_path=checkpoint_prefix, global_step=step)