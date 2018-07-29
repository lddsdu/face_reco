#-*- coding: utf-8 -*-
# @Time     :  下午1:20
# @Author   : ergouzi
# @FileName : data_helper.py
# 从指定的文件中获取到所有的数据
import os
import tensorflow as tf
import cv2
from scipy import misc
import matplotlib.image as mpimg
import numpy as np


data_path = "./lfw_face"
person_file = "person.txt"


def get_image_path_image_label():
    """获取image_path 和 image_label作为统计数据"""
    image_path = []
    image_label = []
    person_id_map = {}
    # 获取到所有的人的图片和label的对应关系
    with open(os.path.join(data_path, person_file)) as f:
        person_all = f.read().split("\n")
        person_all = person_all[:-1]
        for i, person in enumerate(person_all):
            person_id_map[person] = i
        for person in person_all:
            abs_path = os.path.join(data_path, person)
            for img in os.listdir(abs_path):
                image_path.append(os.path.join(abs_path, img))
                image_label.append(person_id_map[person])
    return image_path, image_label



class Dataset(object):

    def __init__(self, train_rate, batch_size):
        self.batch_size = batch_size
        self.num_classes = 5749
        image_path, image_label = get_image_path_image_label()
        self.image_paths = np.asarray(image_path)
        self.image_labels = np.asarray(image_label)

        # shuffle
        new_index = np.random.permutation(len(self.image_labels))
        self.image_paths = self.image_paths[new_index]
        self.image_labels = self.image_labels[new_index]

        # 划分训练和测试数据
        self.train_size = int(len(self.image_labels) * train_rate)
        self.test_size = len(self.image_labels) - self.train_size
        # 数据的地址
        self.train_image_paths = self.image_paths[:self.train_size]
        self.test_image_paths = self.image_paths[self.train_size:]
        # 数据的label
        self.train_image_labels = self.image_labels[: self.train_size]
        self.test_image_labels = self.image_labels[self.train_size:]
        # 数据的data
        self.train_image_data= self.read_images(self.train_image_paths)
        self.test_image_data = self.read_images(self.test_image_paths)
        self.batch_index = 0
        self.batch_test_index = 0

    def read_images(self, image_paths):
        """给定image_path list， 返回对应的图片的numpy的数据表示list"""
        res = []
        for i, img_path in enumerate(image_paths):
            img_data = cv2.imread(img_path)   # 读取数据，类型为numpy.ndarray
            # 原来的图片数据是 250 * 250 的， 现在大小改变为39 31
            img_data = cv2.resize(img_data, (39, 39)) / 255.0
            res.append(img_data[:, 3:34, :])
        return np.asarray(res)

    def next_train_batch(self, batch_size=None):
        """指定batch_size"""
        if batch_size == None:
            batch_size = self.batch_size
        batch_start = self.batch_index
        batch_end = self.batch_index + batch_size
        if batch_end > self.train_size:

            # 重新打乱训练数据
            new_index = np.random.permutation(self.train_size)
            self.train_image_paths = self.train_image_paths[new_index]
            self.train_image_data = self.train_image_data[new_index]
            self.train_image_labels = self.train_image_labels[new_index]
            self.batch_index = 0
            batch_start = self.batch_index
            batch_end = self.batch_index + batch_size
        else:
            self.batch_index += batch_size

        labels = np.zeros([batch_size, self.num_classes], dtype=np.int32)
        for i, index in enumerate(self.train_image_labels[batch_start: batch_end]):
            labels[i][index] = 1

        return labels, self.train_image_data[batch_start:batch_end]

    def next_test_batch(self, batch_size = None):
        if batch_size == None:
            batch_size = self.batch_size
        batch_start = self.batch_test_index
        batch_end = self.batch_test_index + batch_size

        labels = np.zeros([batch_size, self.num_classes], dtype=np.int32)
        for i, index in enumerate(self.train_image_labels[batch_start: batch_end]):
            labels[i][index] = 1

        if batch_size > self.test_size:
            self.batch_test_index = 0
            return False, self.test_image_labels[batch_start: self.test_size],\
                   self.test_image_data[batch_start:batch_end]
        else:
            self.batch_test_index += batch_size
            return True, labels, \
                   self.test_image_data[batch_start: batch_end]