#!/usr/bin/env python
# coding: utf-8

# In[5]:


import numpy as np
import matplotlib.pyplot as plt
import json
import cv2


# In[6]:


data_path = 'physicshub_dir/dfs6/pub/hackathon/DATASETS/SERENGETI/DOWNLOADED'
file = open('SnapshotSerengetiBboxes_20190903.json')
bbox = json.load(file)
file.close()


# In[7]:


def format_bbox_txt(bbox_info, write_index, path, shape):
        img_w, img_h = shape
        cat_id = bbox_info['category_id']
        x,y,w,h = list(bbox_info['bbox'])
        x_center = (x+w/2)/img_w
        y_center = (y+h/2)/img_h
        file = open(f'serengeti/labels/{path}/im{write_index}.txt', 'a+')
        file.write(f'{cat_id} {x_center} {y_center} {w/img_w} {h/img_h}\r\n')
        file.close()

def input_data(bbox_data, path, start=0, end=100):
    annot_list = np.arange(0,len(bbox_data))
    count = 0
    write_index = 0
    im_set = set()
    for index, im_id in enumerate(annot_list[start:end]):
        img_path = bbox_data[im_id]['image_id']
        img_str = img_path.split('/')[-1]
        
        
        img = cv2.imread(f'{data_path}/{img_path}.JPG')
        if type(img)!=type(None):
            size = len(im_set)
            im_set |= {img_str}

            if len(im_set) != size or size==0:
                write_index += 1
                cv2.imwrite(f'serengeti/images/{path}/im{write_index}.JPG', img)

            format_bbox_txt(bbox_data[im_id], write_index, path, (img.shape[1],img.shape[0]) )
        else:
            count+=1


# In[8]:


input_data(bbox['annotations'], 'training', start=0, end=100)
input_data(bbox['annotations'], 'validation', start=100, end=200)
input_data(bbox['annotations'], 'testing', start=200, end=300)


# In[ ]:




