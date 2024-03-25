from calendar import EPOCH
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as Data
import sys
import math
import numpy as np


class CNN(nn.Module):
    def __init__(self, num_conv, num_hidd, layer_scale_init_value=5e-1, spatial_scale_init_value=5e-1):
        super(CNN, self).__init__()  # 继承__init__功能

        # 使用和原CNN一样的卷积操作，包括padding的方式，因为原padding='SMAE'是在前面行补了1行0（上）最后补了2行0（下）
        # 前面补了3列0(左），后面补了4列0（右）,nn.ZeroPad2d(padding_left,right,top,bottom)
        self.pad1 = nn.ZeroPad2d((3, 4, 1, 2))

        '''为了好输出每一步的值的大小, 不用nn.Sequential()搭建'''
        # 第一层卷积
        self.conv1 = nn.Conv2d(6, num_conv, kernel_size=(4, 8), stride=1)
        self.tanh = nn.Tanh()

        self.maxpool = nn.MaxPool2d(kernel_size=2)  # MaxPool2d的stride 不设置就默认为kernel_size的大小

        # 第二层卷积        
        self.pad2 = nn.ZeroPad2d((1, 2, 0, 1))
        self.conv2 = nn.Conv2d(num_conv, num_conv, kernel_size=(2, 4), stride=1)

        # 第三层卷积、没有maxpooling            
        self.pad3 = nn.ZeroPad2d((1, 2, 0, 1))  # 需要补的情况和pad2一样
        self.conv3 = nn.Conv2d(num_conv, num_conv, kernel_size=(2, 4), stride=1)

        # 全连接，nn.Linear的输入是应该是二维张量（batch, infeatures）
        self.linear = nn.Linear(num_conv * 6 * 18, num_hidd)

        ## 输出层
        self.output = nn.Linear(num_hidd, 1)

        '''spatial scale'''
        # spatial scale conv1
        self.ssconv1 = nn.Parameter(spatial_scale_init_value * torch.ones((24, 72)),
                                    requires_grad=True) if spatial_scale_init_value > 0 else None
        print('Spatial Scale conv1:', self.ssconv1.shape, self.ssconv1)
        # spatial scale conv2
        self.ssconv2 = nn.Parameter(spatial_scale_init_value * torch.ones((12, 36)),
                                    requires_grad=True) if spatial_scale_init_value > 0 else None
        print('Spatial Scale conv2:', self.ssconv2.shape, self.ssconv2)
        # spatial scale conv3
        self.ssconv3 = nn.Parameter(spatial_scale_init_value * torch.ones((6, 18)),
                                    requires_grad=True) if spatial_scale_init_value > 0 else None
        print('Spatial Scale conv3:', self.ssconv3.shape, self.ssconv3)

        '''layer scale'''
        # layer scale conv1
        self.lsconv1 = nn.Parameter(layer_scale_init_value * torch.ones((num_conv,)),
                                    requires_grad=True) if layer_scale_init_value > 0 else None
        print('Layer Scale conv1:', self.lsconv1)
        # layer scale conv2
        self.lsconv2 = nn.Parameter(layer_scale_init_value * torch.ones((num_conv,)),
                                    requires_grad=True) if layer_scale_init_value > 0 else None
        print('Layer Scale conv2:', self.lsconv2)
        # layer scale conv3
        self.lsconv3 = nn.Parameter(layer_scale_init_value * torch.ones((num_conv,)),
                                    requires_grad=True) if layer_scale_init_value > 0 else None
        print('Layer Scale conv3:', self.lsconv3)
        # gammalinear
        self.gammalinear = nn.Parameter(layer_scale_init_value * torch.ones((num_hidd,)),
                                        requires_grad=True) if layer_scale_init_value > 0 else None
        print('Layer Scale linear:', self.gammalinear)

        '''调用初始化'''
        self.apply(self._init_weights)

    def truncated_normal_(self, tensor, mean=0, std=0.02):
        # 有的pytorch版本中没有这个初始化方法，就定义一个类似的
        with torch.no_grad():
            size = tensor.shape
            tmp = tensor.new_empty(size + (4,)).normal_()
            valid = (tmp < 2) & (tmp > -2)
            ind = valid.max(-1, keepdim=True)[1]
            tensor.data.copy_(tmp.gather(-1, ind).squeeze(-1))
            tensor.data.mul_(std).add_(mean)
            return tensor

    def _init_weights(self, m):
        # 如果子模块是卷积或全连接，就对权重和偏置进行相应初始化
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            # self.truncated_normal_(m.weight, std=0.2)#nn.init.trunc_normal_((m.weight, std=0.2)低版本没有这个函数
            # nn.init.normal_(m.weight,mean=0,std=1)#使用原CNN正态分布对权重初始化
            nn.init.trunc_normal_(m.weight, std=0.2)
            # nn.init.xavier_uniform_(m.weight.data,gain=1) #xavier均匀分布
            # nn.init.xavier_normal_(m.weight.data,gain=1)  #xavier正态分布

            nn.init.constant_(m.bias, 0)

    def forward(self, x):

        # 第一层
        x = self.pad1(x)
        x = self.conv1(x)  # [batch,num_conv,24,72]
        print('before tanh1 min and max:', x.min(), x.max())

        if self.ssconv1 is not None:

            x = self.ssconv1 * x  # Layer Scale 对通道数值进行放缩, * 对应最后一维相同
            # print('ssconv1 min max:',self.ssconv1.min(),self.ssconv1.max())

            if self.lsconv1 is not None:
                x = x.permute(0, 2, 3, 1)  # [N, C, H, W] -> [N, H, W, C] 为了对通道加权相乘，最后一维要相同
                x = self.lsconv1 * x
                # print('lsconv1 min max:',self.lsconv1.min(),self.lsconv1.max())
                x = x.permute(0, 3, 1, 2)  # [N, H, W, C] -> [N, C, H, W] 还原回去

        x = self.tanh(x)
        print('tanh1 min and max:', x.min(), x.max())

        # x1 = x.detach().cpu().numpy()
        # print('conv1 all:',x1.size,'+1:',np.sum(x1 >= 0.99),'-1:',np.sum(x1 <= -0.99))
        # print('tanh1 1and-1 ratio:',round((np.sum(x1 >= 0.99)+np.sum(x1 <= -0.99)) / x1.size,3))#保留3位小数

        x = self.maxpool(x)

        # 第二层
        x = self.pad2(x)
        x = self.conv2(x)  # [batch,num_conv,12,36]
        # print('before tanh2 min and max:',x.min(),x.max())

        if self.ssconv2 is not None:

            x = self.ssconv2 * x  # Layer Scale 对通道数值进行放缩, * 对应最后一维相同
            # print('ssconv2 min max:',self.ssconv2.min(),self.ssconv2.max())

            if self.lsconv2 is not None:
                x = x.permute(0, 2, 3, 1)  # [N, C, H, W] -> [N, H, W, C] 为了对通道加权相乘，最后一维要相同
                x = self.lsconv2 * x
                # print('lsconv2 min max:',self.lsconv2.min(),self.lsconv2.max())
                x = x.permute(0, 3, 1, 2)  # [N, H, W, C] -> [N, C, H, W] 还原回去

        x = self.tanh(x)
        print('tanh2 min and max:', x.min(), x.max())

        # x1 = x.detach().cpu().numpy()
        # print('conv2 all:',x1.size,'+1:',np.sum(x1 >= 0.99),'-1:',np.sum(x1 <= -0.99))
        # print('tanh2 1and-1 ratio:',round((np.sum(x1 >= 0.99)+np.sum(x1 <= -0.99)) / x1.size,3))

        x = self.maxpool(x)

        # 第三层
        x = self.pad3(x)
        x = self.conv3(x)
        # print('before tanh3 min and max:',x.min(),x.max())

        if self.ssconv3 is not None:

            x = self.ssconv3 * x  # Layer Scale 对通道数值进行放缩, * 对应最后一维相同
            # print('ssconv3 min max:',self.ssconv3.min(),self.ssconv3.max())

            if self.lsconv3 is not None:
                x = x.permute(0, 2, 3, 1)  # [N, C, H, W] -> [N, H, W, C] 为了对通道加权相乘，最后一维要相同
                x = self.lsconv3 * x
                # print('lsconv3 min max:',self.lsconv3.min(),self.lsconv3.max())
                x = x.permute(0, 3, 1, 2)  # [N, H, W, C] -> [N, C, H, W] 还原回去

        x = self.tanh(x)
        print('tanh3 min and max:', x.min(), x.max())

        # x1 = x.detach().cpu().numpy()
        # print('conv3 all:',x1.size,'+1:',np.sum(x1 >= 0.99),'-1:',np.sum(x1 <= -0.99)) 
        # print('tanh3 1and-1 ratio:',round((np.sum(x1 >= 0.99)+np.sum(x1 <= -0.99)) / x1.size,3))

        x = x.view(x.size(0), -1)  # 保留batch, 将后面的乘到一起 [batch, num_conv * 6 *18]

        # 全连接、全连接之后也经过了tanh
        x = self.linear(x)
        # print('before linear tanh min and max:',x.min(),x.max())

        if self.gammalinear is not None:
            x = self.gammalinear * x  # Layer Scale 对通道数值进行放缩, * 对应最后一维相同
            # print('gammalinear min max:',self.gammalinear.min(),self.gammalinear.max())

        x = self.tanh(x)
        print('linear tanh min and max:', x.min(), x.max())

        x1 = x.detach().cpu().numpy()
        print('linear all:', x1.size, '+1:', np.sum(x1 >= 0.99), '-1:', np.sum(x1 <= -0.99))
        print('linear tanh 1and-1 ratio:', round((np.sum(x1 >= 0.99) + np.sum(x1 <= -0.99)) / x1.size, 3))

        output = self.output(x)  # 输出[batch,1]，要压缩到1维，否则会警告
        output = output.squeeze(-1)

        '''要压缩到1维, torch.Size([161])) that is different to the input size (torch.Size([161, 1])). 
        This will likely lead to incorrect results due to broadcasting'''

        return output


class MyDataSet(Data.Dataset):
    """自定义DataSet"""

    def __init__(self, inputs, lable):
        super(MyDataSet, self).__init__()
        self.inputs = inputs
        self.lable = lable

    """定义mydataset类都需要定义len和getitem两个魔法函数,len返回数据集的长度,getitem返回数据和标签"""

    def __len__(self):
        return self.inputs.shape[0]

    """idx,index是一个索引,这个索引的取值范围是要根据__len__这个返回值确定的"""

    def __getitem__(self, index):
        return self.inputs[index], self.lable[index]


def train_one_epoch_noupdate_lr(model, optimizer, data_loader, device, epoch):
    model.train()
    loss_function = torch.nn.MSELoss()  # reduction='sum' or 'mean'
    accu_loss = torch.zeros(1).to(device)  # 累计损失

    optimizer.zero_grad()
    # epoch之间梯度置0,不让之前epoch的梯度影响这个epoch的梯度，也可以放在下个循环里loss.backward之前

    for step, data in enumerate(data_loader):
        # enumerate返回值有两个，一个是序号（batch的地址,从几开始）一个是数据data（包括训练数据和标签）
        # enumerate(data_loader,1)表示batch的地址从1开始，比如总共5个batch，那么序号（step）就是1-5，不是0-4
        # 不用enumerate直接输入没有batch序号的数据，会导致寻址的时候报错
        print('step:', step)
        datacmip, labels = data
        datacmip, labels = datacmip.to(device), labels.to(device)

        pred = model(datacmip)  # 输入数据正向传播得到预测

        loss = loss_function(pred, labels)
        loss.backward()
        accu_loss += loss.detach()  # 返回的仍是tensor,但是不会去计算其梯度
        # detach()操作后的tensor与原始tensor共享数据内存，c=a.detach()就是c随着a变化而变化
        # 当原始tensor在计算图中数值发生反向传播等更新之后，detach()的tensor值也发生了改变
        # 同时不要直接修改loss.detach，因为和loss共用一个内存，修改了之后会对求梯度报错

        print("[train epoch {}], batch mean loss: {:.8f}, lr: {:.8f}".format(
            epoch,
            accu_loss.item() / (step + 1),
            optimizer.param_groups[0]["lr"]))
        # optimizer.param_groups：是长度为2的list，其中的元素是2个字典,包含lr，weight_decay等参数
        # 填充到{}里,这里loss是和之前batch平均的loss结果
        # .item()返回的是一个浮点型数据,原来的loss是一个可优化的tensor变量，
        # 不能直接加减操作，否则会被认为是动态图中另外的新变量，计算图（网络）不断增大

        if not torch.isfinite(loss):  # 判断loss是否有界true或者false
            print('WARNING: non-finite loss, ending training ', loss)
            sys.exit(1)  # 退出程序，非0值意为异常终止

        optimizer.step()  # 优化器进行一次参数优化
        # 放在每一个batch训练而不是一个epoch，这是因为是将每一次mini-batch看作一次训练，更新一次参数

        optimizer.zero_grad()
        # 一个epoch内，不同batch之间梯度置零，如果不清零，那么使用的这个grad就会和上一个mini-batch有关

    return accu_loss.item() / (step + 1)


def train_one_epoch(model, optimizer, data_loader, device, epoch, lr_scheduler):
    model.train()
    loss_function = torch.nn.MSELoss()  # reduction='sum' or 'mean'
    accu_loss = torch.zeros(1).to(device)  # 累计损失

    optimizer.zero_grad()  # 梯度初始化置0

    for step, data in enumerate(data_loader):
        # enumerate返回值有两个，一个是序号（batch的地址,从几开始）一个是数据data（包括训练数据和标签）
        # enumerate(data_loader,1)表示batch的地址从1开始，比如总共5个batch，那么序号（step）就是1-5，不是0-4
        # 不用enumerate直接输入没有batch序号的数据，会导致寻址的时候报错
        print('step:', step)
        datacmip, labels = data
        datacmip, labels = datacmip.to(device), labels.to(device)

        pred = model(datacmip)  # 输入数据正向传播得到预测

        loss = loss_function(pred, labels)
        loss.backward()
        accu_loss += loss.detach()  # 返回的仍是tensor,但是不会去计算其梯度
        # detach()操作后的tensor与原始tensor共享数据内存，c=a.detach()就是c随着a变化而变化
        # 当原始tensor在计算图中数值发生反向传播等更新之后，detach()的tensor值也发生了改变
        # 同时不要直接修改loss.detach，因为和loss共用一个内存，修改了之后会对求梯度报错

        print("[train epoch {}], batch mean loss: {:.8f}, lr: {:.8f}".format(
            epoch,
            accu_loss.item() / (step + 1),
            optimizer.param_groups[0]["lr"]))  # 填充到{}里,这里loss是和之前batch平均的loss结果
        # .item()返回的是一个浮点型数据,原来的loss是一个可优化的tensor变量，
        # 不能直接加减操作，否则会被认为是动态图中另外的新变量，计算图（网络）不断增大

        if not torch.isfinite(loss):  # 判断loss是否有界true或者false
            print('WARNING: non-finite loss, ending training ', loss)
            sys.exit(1)  # 退出程序，非0值意为异常终止

        optimizer.step()  # 优化器进行一次参数优化
        # 放在每一个batch训练而不是一个epoch，这是因为是将每一次mini-batch看作一次训练，更新一次参数

        optimizer.zero_grad()
        # 如果不清零，那么使用的这个grad就会和上一个mini-batch有关
        # update lr
        lr_scheduler.step()

    return accu_loss.item() / (step + 1)


def create_lr_scheduler(optimizer,
                        num_step: int,
                        epochs: int,
                        warmup=True,
                        warmup_epochs=1,
                        warmup_factor=1e-3,
                        end_factor=1e-6):
    assert num_step > 0 and epochs > 0
    if warmup is False:
        warmup_epochs = 0
    # num_step与batchsize大小有关，就是多少个batch的数量，一次迭代需要分几个step
    '''Warmup是在ResNet论文中提到的一种学习率预热的方法，它在训练开始的时候先选择使用一个较小的学习率，
    训练了一些epoches或者steps(比如4个epoches,10个steps),再修改为预先设置的学习率来进行训练
    如果warmup=false，那就直接使用余弦退火学习率调整，不先预热学习率'''

    def f(x):
        """
        x应该是预设固定的学习率
        根据step数返回一个学习率倍率因子，
        注意在训练开始之前，pytorch会提前调用一次lr_scheduler.step()方法
        """
        if warmup is True and x <= (warmup_epochs * num_step):
            alpha = float(x) / (warmup_epochs * num_step)
            # warmup过程中lr倍率因子从warmup_factor -> 1
            return warmup_factor * (1 - alpha) + alpha
        else:
            current_step = (x - warmup_epochs * num_step)
            cosine_steps = (epochs - warmup_epochs) * num_step
            # warmup后lr倍率因子从1 -> end_factor
            return ((1 + math.cos(current_step * math.pi / cosine_steps)) / 2) * (1 - end_factor) + end_factor
        # 余弦退火学习率调整策略

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=f)
    # pytorch提供的自定义调整学习率策略 LambdaLR 可以自己制定规则调整学习率
    # lr_lambda传入自定义的函数或lambda表达式，可以对Optimizer中的不同的param_groups制定不同的调整规则
    # 传入的lr_lambda参数会在梯度下降时对optimizer对应参数组的学习率乘上一个权重系数


def create_lr_scheduler(optimizer,
                        num_step: int,
                        epochs: int,
                        warmup=True,
                        warmup_epochs=1,
                        warmup_factor=1e-3,
                        end_factor=1e-6):
    assert num_step > 0 and epochs > 0
    if warmup is False:
        warmup_epochs = 0
    # num_step与batchsize大小有关，就是多少个batch的数量，一次迭代需要分几个step
    '''Warmup是在ResNet论文中提到的一种学习率预热的方法，它在训练开始的时候先选择使用一个较小的学习率，
    训练了一些epoches或者steps(比如4个epoches,10个steps),再修改为预先设置的学习率来进行训练
    如果warmup=false，那就直接使用余弦退火学习率调整，不先预热学习率'''

    def f(x):
        """
        x应该是预设固定的学习率
        根据step数返回一个学习率倍率因子，
        注意在训练开始之前，pytorch会提前调用一次lr_scheduler.step()方法
        """
        if warmup is True and x <= (warmup_epochs * num_step):
            alpha = float(x) / (warmup_epochs * num_step)
            # warmup过程中lr倍率因子从warmup_factor -> 1
            return warmup_factor * (1 - alpha) + alpha
        else:
            current_step = (x - warmup_epochs * num_step)
            cosine_steps = (epochs - warmup_epochs) * num_step
            # warmup后lr倍率因子从1 -> end_factor
            return ((1 + math.cos(current_step * math.pi / cosine_steps)) / 2) * (1 - end_factor) + end_factor
        # 余弦退火学习率调整策略

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=f)
    # pytorch提供的自定义调整学习率策略 LambdaLR 可以自己制定规则调整学习率
    # lr_lambda传入自定义的函数或lambda表达式，可以对Optimizer中的不同的param_groups制定不同的调整规则
    # 传入的lr_lambda参数会在梯度下降时对optimizer对应参数组的学习率乘上一个权重系数
