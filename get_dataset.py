import os
import re
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms


#自定义输入数据集
class my_dataset(Dataset):

    def __init__(self,root,label_dir,data_tranform=None):
        super(my_dataset,self).__init__()
        imgs=[]  #用来保存图片的地址
        label=[]  #保存标签的地址

        #得到图片路径
        root_list=os.listdir(root)
        for i in range(len(root_list)):
            imgs.append(root+'/'+str(i+1)+'.jpg')   #组合路径

        #得到标签
        label_file=open(label_dir,'r')
        label_list=label_file.readlines()
        for i in label_list:
            label.append(int(i))

        self.imgs=imgs
        self.label=label
        self.tranform=data_tranform

        # img=Image.open(imgs[0])
        # img.show()
        #
        # print(imgs)
        # print(len(label))

    def __getitem__(self, item):
        #print(item)  #查看现在数据加载的进度
        the_imag=self.imgs[item]  #这里得到的是路径
        the_label=self.label[item]   #得到标签

        the_imag=Image.open(the_imag).convert('RGB')  #把路径转化为图片
        if self.tranform is not None:
            the_imag=self.tranform(the_imag)  #处理图片
        return the_imag,the_label

    def __len__(self):
        return len(self.imgs)

