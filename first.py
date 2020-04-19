import numpy as np
import torch
from torch.autograd import Variable
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import torch.nn.functional as F
from get_dataset import my_dataset

# hyper parameter
BATCH_SIZE = 50
EPOCH = 15

# 转变
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

#训练集
trainset = my_dataset('./train_img', './train_label.txt', transform)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE,
                                          shuffle=False, num_workers=0)
#测试集
testset = my_dataset('./test_img', './test_label.txt', transform)

testloader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE,
                                          shuffle=False, num_workers=0)

class CNN_NET(torch.nn.Module):
    def __init__(self):
        super(CNN_NET, self).__init__()
        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv2d(3, 32, 3, 1, 1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2))
        self.conv2 = torch.nn.Sequential(
            torch.nn.Conv2d(32, 64, 3, 1, 1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2)  #kernel为2，stride也是
       )
        # self.conv3 = torch.nn.Sequential(
        #     torch.nn.Conv2d(64, 64, 3, 1, 1),
        #     torch.nn.ReLU(),
        #     torch.nn.MaxPool2d(2)
        # )
        self.dense = torch.nn.Sequential(
            torch.nn.Linear(64 * 4 * 4, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 12)
        )

    def forward(self, x):
        conv1_out = self.conv1(x)
        conv2_out = self.conv2(conv1_out)
        #conv3_out = self.conv3(conv2_out)
        res = conv2_out.view(conv2_out.size(0), -1)
        out = self.dense(res)
        return out


# net = CNN_NET()
# print(net)
#
# # 损失与优化函数
# import torch.optim as optim
#
# optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
# loss_func = torch.nn.CrossEntropyLoss()
#
# for epoch in range(EPOCH):
#     print('epoch {}'.format(epoch + 1))
#     running_loss = 0.0
#     train_acc=0.0
#     for step, data in enumerate(trainloader):
#         print('step: %d' % step)   #查看训练进度
#
#         b_x, b_y = data    #图片与标签的输入
#         b_x,b_y=Variable(b_x),Variable(b_y)
#         outputs = net.forward(b_x)  # 神经网络输入
#         loss = loss_func(outputs, b_y)  # 计算误差函数
#
#         #统计预测对的个数
#         pred=torch.max(outputs,1)[1]  #输出结果转化为数字
#         train_corrent=(pred==b_y).sum()
#         train_acc+=train_corrent.data
#
#         #反向传播
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
#         # 打印状态信息
#         running_loss += loss.item()
#         # if step % 1000 == 999:  # 每2000批次打印一次
#     #每一次epoch打印一次
#     torch.save(net,'./first_model.pkl')
#     print('[%d, %5d] loss: %.3f  acc: %.3f' % (epoch + 1, step + 1, running_loss / len(trainset),train_acc/len(trainset)))
#
# print('Finished Training')
#
#测试
net=torch.load('./first_model.pkl').eval()
eval_loss = 0.
eval_acc = 0.
i=0
for batch_x, batch_y in testloader:
    print(i)
    batch_x, batch_y = Variable(batch_x, volatile=True), Variable(batch_y, volatile=True)
    out = net(batch_x)
    # loss = loss_func(out, batch_y)
    # eval_loss += loss.data[0]
    pred = torch.max(out, 1)[1]
    num_correct = (pred == batch_y).sum()
    eval_acc += num_correct.data
    i=i+1
print('acc  %.3f' % (eval_acc/len(testset)))