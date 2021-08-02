import tensorflow as tf
import random
import os
import re

def image_deals1(train_file):       # 读取原始文件
    image_string = tf.io.read_file(train_file)  # 读取原始文件
    image_decoded = tf.image.decode_jpeg(image_string)  # 解码JPEG图片
    image_decoded=randoc(image_decoded)
    image_decoded= tf.image.resize(image_decoded, [299, 299])  #把图片转换为224*224的大小
    #image = tf.image.rgb_to_grayscale(image_decoded)
    image = tf.cast(image_decoded, dtype=tf.float32) / 255.0-0.5
    return image
def image_deals(train_file):       # 读取原始文件
    image_string = tf.io.read_file(train_file)  # 读取原始文件
    image_decoded = tf.image.decode_jpeg(image_string)  # 解码JPEG图片
    image_decoded=randoc(image_decoded)
    image_decoded= tf.image.resize(image_decoded, [299, 299])  #把图片转换为224*224的大小
    #image = tf.image.rgb_to_grayscale(image_decoded)
    image = tf.cast(image_decoded, dtype=tf.float32) / 255.0-0.5
    return image
def randoc(train_file):
    int1=random.randint(1,10)
    if int1==1:
        train_file = tf.image.random_flip_left_right(train_file)   #左右翻折
    elif int1==2:
        train_file=tf.image.random_flip_up_down(train_file)
    return train_file

def train_test_get(train_test_inf):
    for root,dir,files in os.walk(train_test_inf, topdown=False):
        #print(root)
        #print(files)
        list1=[root+"/"+i for i in files]
        return list1
def class_get():
    json_train=train_test_get("C:/Users/mzy/Desktop/机器学习/dddd/keypoint_image_part1.tar/cat")
    json_train1=train_test_get("C:/Users/mzy/Desktop/机器学习/dddd/keypoint_image_part1.tar/cow")
    json_train2=train_test_get("C:/Users/mzy/Desktop/机器学习/dddd/keypoint_image_part1.tar/dog")
    json_train3 = train_test_get("C:/Users/mzy/Desktop/机器学习/dddd/keypoint_image_part1.tar/sheep")
    json_train4=train_test_get("C:/Users/mzy/Desktop/机器学习/dddd/keypoint_image_part1.tar/horse")
    list1=[0 for i in range(len(json_train))]
    list2 = [1 for i in range(len(json_train1))]
    list3 = [2 for i in range(len(json_train2))]
    list4 = [3 for i in range(len(json_train3))]
    list5 = [4 for i in range(len(json_train4))]
    list_label=list1+list2+list3+list4+list5
    train=json_train+json_train1+json_train2+json_train3+json_train4
    list_label=tf.one_hot(list_label,depth=5)
    image_list=[image_deals(i) for i in train]
    dataest=tf.data.Dataset.from_tensor_slices((image_list, list_label))
    dataest=dataest.shuffle(buffer_size=300).prefetch(tf.data.experimental.AUTOTUNE).repeat(10).batch(100)
    return dataest
#daseaset=rubeish_get()
#print(daseaset)
def class_get1():
    json_train = train_test_get("C:/Users/mzy/Desktop/机器学习/dddd/keypoint_image_part1.tar/cat1")
    json_train1 = train_test_get("C:/Users/mzy/Desktop/机器学习/dddd/keypoint_image_part1.tar/cow1")
    json_train2 = train_test_get("C:/Users/mzy/Desktop/机器学习/dddd/keypoint_image_part1.tar/dog1")
    json_train3 = train_test_get("C:/Users/mzy/Desktop/机器学习/dddd/keypoint_image_part1.tar/sheep1")
    json_train4 = train_test_get("C:/Users/mzy/Desktop/机器学习/dddd/keypoint_image_part1.tar/horse1")
    list1 = [0 for i in range(len(json_train))]
    list2 = [1 for i in range(len(json_train1))]
    list3 = [2 for i in range(len(json_train2))]
    list4 = [3 for i in range(len(json_train3))]
    list5 = [4 for i in range(len(json_train4))]
    list_label = list1 + list2 + list3 + list4 + list5
    train = json_train + json_train1 + json_train2 + json_train3 + json_train4
    list_label = tf.one_hot(list_label, depth=5)
    print(list_label)
    image_list = [image_deals(i) for i in train]
    dataest = tf.data.Dataset.from_tensor_slices((image_list, list_label))
    dataest = dataest.shuffle(buffer_size=300).prefetch(tf.data.experimental.AUTOTUNE).repeat(10).batch(1)
    return dataest
