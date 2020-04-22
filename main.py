import numpy as np
import torch
from torch.autograd import Variable
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import torch.nn.functional as F
from get_dataset import my_dataset


from nets import vgg_net
from nets import my_cnn
from nets import alex_net

# hyper parameter
BATCH_SIZE = 50
EPOCH = 5
LR=0.001
acc_file= './my_cnn_acc_file.txt'

# 转变
train_transform = transforms.Compose([
    #transforms.RandomCrop(16, padding=2),  #随即剪切
    #transforms.RandomHorizontalFlip(),  # 图像一半的概率翻转，一半的概率不翻转
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

#训练集
trainset = my_dataset('./train_img', './train_label.txt', train_transform)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE,
                                          shuffle=True, num_workers=0)
#测试集
testset = my_dataset('./test_img', './test_label.txt', test_transform)

testloader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE,
                                          shuffle=True, num_workers=0)
#
net = my_cnn.CNN_NET()
#net=vgg_net.VGGNeT()
#net=alex_net.AlexNet()
print(net)

# 损失与优化函数
import torch.optim as optim

optimizer = optim.SGD(net.parameters(), lr=LR, momentum=0.9)
loss_func = torch.nn.CrossEntropyLoss()

for epoch in range(EPOCH):
    print('epoch {}'.format(epoch + 1))
    running_loss = 0.0
    train_acc=0.0  #这个epoch总的acc
    this_step_train_acc=0.0  #每200个step打印一个准确率
    for step, data in enumerate(trainloader):
        print('step: %d' % step)   #查看训练进度

        b_x, b_y = data    #图片与标签的输入
        b_x,b_y=Variable(b_x),Variable(b_y)
        outputs = net.forward(b_x)  # 神经网络输入
        loss = loss_func(outputs, b_y)  # 计算误差函数

        #统计预测对的个数
        pred=torch.max(outputs,1)[1]  #输出结果转化为数字
        train_corrent=(pred==b_y).sum()
        train_acc+=train_corrent.data   #这个epoch总的acc
        this_step_train_acc+=train_corrent.data

        #反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # 打印状态信息
        running_loss += loss.item()
        if step % 100 == 99:  # 每100批次打印一次准确率
            print('[%d, %5d]   acc: %.3f' % (epoch + 1, step + 1,this_step_train_acc/(100*BATCH_SIZE)))
            # with open(acc_file,'a+') as file:
            #     file.write(str(this_step_train_acc.float()/(100*BATCH_SIZE))+'\n')
            this_step_train_acc = 0.0
    #每一次epoch打印一次,并且保存一次网络
    torch.save(net,'./my_cnn_model(1).pkl')
    print('[%d, %5d] loss: %.3f  acc: %.3f' % (epoch + 1, step + 1, running_loss / len(trainset),train_acc/len(trainset)))

print('Finished Training')

# #测试
# nets=torch.load('./vgg_net_model.pkl').eval()
# eval_loss = 0.
# eval_acc = 0.
# i=0
# for batch_x, batch_y in testloader:
#     print(i)
#     batch_x, batch_y = Variable(batch_x, volatile=True), Variable(batch_y, volatile=True)
#     out = nets(batch_x)
#     # loss = loss_func(out, batch_y)
#     # eval_loss += loss.data[0]
#     pred = torch.max(out, 1)[1]
#     num_correct = (pred == batch_y).sum()
#     eval_acc += num_correct.data
#     i=i+1
# print('acc  %.3f' % (eval_acc/len(testset)))