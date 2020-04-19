import csv
from PIL import Image
import numpy as np

filename='./train.csv'  #处理的目标文件
label_file = './train_label.txt'   #标签文件
root='./train_img/'    #处理后的图片放在的文件夹


all_class=["WWW","MAIL","FTP-CONTROL","FTP-PASV","ATTACK","P2P","DATABASE","FTP-DATA","MULTIMEDIA","SERVICES","INTERACTIVE","GAMES"]
#用来存储每一列的最大最小值
max=[]
min=[]

with open(filename,encoding='UTF-8') as f:
    reader=csv.reader(f)
    list_reader=list(reader)   #转化为list更好处理
    del list_reader[0]  #删除第一行无用的数据
    #只使用后二十万个数据,便于读取
    get_list=list_reader[-200000:]
    #数据的个数
    data_num=len(get_list)

    #先将最后一列(类别)数据转化成数字
    for i in range(data_num):
        class_name=get_list[i][-1]
        get_list[i][-1]=all_class.index(class_name)+1

    #将小数转化为数字,并且在[0,255归一化]
    for i in range(len(get_list)):
        for j in range(len(get_list[0])):
            if get_list[i][j]=='?':
                get_list[i][j]=2
            elif get_list[i][j]=='Y':
                get_list[i][j]=1
            elif get_list[i][j]=='N':
                get_list[i][j]=0
            else:
                get_list[i][j]=float(get_list[i][j])  #str转化为数字

    #得到每一列的最大最小值
    for i in range(len(get_list[0])):
        min_num=999999
        max_num=0
        for j in range(len(get_list)):
            if get_list[j][i]>max_num:
                max_num=get_list[j][i]
            if get_list[j][i]<min_num:
                min_num=get_list[j][i]
        min.append(min_num)
        max.append(max_num)

    #归一化
    for i in range(len(get_list[0])):
        for j in range(len(get_list)):
            cha=max[i]-min[i]  #要判断一列的数据是不是都是相同的
            if cha!=0:
                k=255/cha
                get_list[j][i]=int(k*(get_list[j][i]-min[i]))
            else:
                get_list[j][i]=int(get_list[j][i])

    #补全到256位
    for i in range(len(get_list)):
        for j in range(7):
            get_list[i].append(0)

    # #看一下标签都是什么
    # for i in range(len(get_list)):
    #     print(get_list[i][-8])  #这个位置就是标签的所在

    #保存为图片，同时保存标签

    for i in range(len(get_list)):
        print(i)   #看一下进度
        with open(label_file,'a+') as label:
            test_img_arr=np.array(get_list[i]).reshape(16,16)   #转化为16*16的图片
            picture = Image.fromarray(test_img_arr).convert("RGB")  #转化成rgb便于神经网络输入
            picture.save(root+str(i+1)+'.jpg')  #保存图片
            label.write(str(int(get_list[i][-8]/23))+'\n')   #保存标签并且换行显示

