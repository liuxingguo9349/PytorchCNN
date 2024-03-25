from pyexpat import model
from zmq import device
import torch
import torch.nn as nn
import torch.utils.data as Data
import copy
import os
import math
import numpy as np
from netCDF4 import Dataset

from model import CNN,train_one_epoch,train_one_epoch_noupdate_lr,MyDataSet,create_lr_scheduler

devices = 'cuda:0'
epochs = 10              #原cnn20代
batch_size = 20
learning_rate = 5e-4    #原cnn学习率0.005
num_conv = 3
num_hidd = 3

lead_mon = 6
target_mon = 1
tg_mn = int(target_mon - 1)
ld_mn1 = int(23 - lead_mon + tg_mn)
ld_mn2 = int(23 - lead_mon + tg_mn + 3)
# Read Data (NetCDF4)
inp1 = Dataset('/content/drive/MyDrive/input/SODA.input.36mon.1871_1970.nc','r')
inp2 = Dataset('/content/drive/MyDrive/input/SODA.label.12mon.1873_1972.nc','r')

inpv1 = np.zeros((100,6,24,72),dtype=np.float32)
inpv1[:,0:3,:,:] = inp1.variables['sst'][0:100,ld_mn1:ld_mn2,:,:]
inpv1[:,3:6,:,:] = inp1.variables['t300'][0:100,ld_mn1:ld_mn2,:,:]

inpv2 = np.zeros((100),dtype=np.float32)
inpv2[:] = inp2.variables['pr'][0:100,tg_mn,0,0]

'''测试集'''
test = Dataset('/content/drive/MyDrive/input/GODAS.input.36mon.1980_2015.nc','r')

test1 = np.zeros((36,6,24,72),dtype=np.float32)
test1[:,0:3,:,:] = test.variables['sst'][0:36,ld_mn1:ld_mn2,:,:]
test1[:,3:6,:,:] = test.variables['t300'][0:36,ld_mn1:ld_mn2,:,:]


testlable = Dataset('/content/drive/MyDrive/input/GODAS.label.12mon.1982_2017.nc', 'r')


def main():

    device = torch.device(devices if torch.cuda.is_available() else "cpu")
    print(f"using {device} device.")


    #实例化数据集
    train_dataset = MyDataSet(inpv1,inpv2)

    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers多进程加载的进程数，0代表单进程
    print('Using {} dataloader workers every process'.format(nw))
   
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               pin_memory=False,
                                               num_workers=nw,
                                               )#collate_fn=train_dataset.collate_fn
    #pin_memory=true可以更快的将tensor转到GPU，但占用的内存较多
    #drop_last：告诉如何处理数据集长度除于batch_size余下的数据。默认为false，True就抛弃，false保留，剩多少用多少
    #若不设置collate_fn参数则会使用默认处理函数 但必须保证传进来的数据都是tensor格式否则会报错
    #可以试试不设置也就是默认collate_fn
    """collate_fn可以在调用__getitem__函数后,将得到的batch_size个数据进行进一步的处理,
    在迭代dataloader时,取出的数据批就是经过了collate_fn函数处理的数据。
    换句话说,collate_fn的输入参数是__getitem__的返回值,dataloader的输出是collate_fn的返回值。"""
    
    model = CNN(num_conv=num_conv, num_hidd=num_hidd).to(device)
    bestmodel ='/content/drive/MyDrive/cmip/bestcmip_model.pth'
    if os.path.exists(bestmodel) is False:
        bestmodel = '/content/drive/MyDrive/cmip/last_model.pth'

    weights_dict = torch.load(bestmodel, map_location=device)#["model"]

    print(model.load_state_dict(weights_dict, strict=True))
    #加载模型model.load_state_dict(torch.load(PATH)
    #strict=默认为true，参数依次赋值给新模型，当新旧模型不完全一致就会报错，
    #False就是只将key一致的变量，才进行加载赋值
    #torch.load是加载训练好的模型，load_state_dict是net的方法，将torch.load加载出来的数据加载到net中


    #alpha=0.9 衰减（平滑）系数，默认0.99，改为和tf一样0.9
    optimizer = torch.optim.RMSprop(model.parameters(), lr=learning_rate, alpha=0.9,eps=1e-10)
    #optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)


    #学习率，根据训练次数调整
    #lr_scheduler = create_lr_scheduler(optimizer, len(train_loader), epochs,
    #                                   warmup=True, warmup_epochs=1)

    many = epochs // 2
    grad = np.zeros((many,6,24,72))
    cor = np.zeros((epochs))
    bestloss = float("inf")
    testloss = np.zeros((epochs))

    #打印训练的参数和梯度情况
    for name, param in model.named_parameters():
        print(name,param.requires_grad)


    for epoch in range(epochs):
        print('grad shape:',grad.shape)
        # train
        """train_loss = train_one_epoch(model=model,
                                    optimizer=optimizer,
                                    data_loader=train_loader,
                                    device=device,
                                    epoch=epoch,
                                    lr_scheduler=lr_scheduler)"""    

        train_loss = train_one_epoch_noupdate_lr(model=model,
                                    optimizer=optimizer,
                                    data_loader=train_loader,
                                    device=device,
                                    epoch=epoch)

        if epoch == 0:
            
            torch.save(copy.deepcopy(model.state_dict()), '/content/drive/MyDrive/ENnumber/first_model.pth')
        
        print('train_loss:',train_loss)
        if train_loss < bestloss:
            
            torch.save(copy.deepcopy(model.state_dict()), '/content/drive/MyDrive/ENnumber/best_model.pth')#只保存权重,state_dict只是浅拷贝
            bestloss = train_loss
            print('bestloss:',bestloss)
        else:
            print('this loss is not the best')
            
        torch.save(model.state_dict(), '/content/drive/MyDrive/ENnumber/last_model.pth')


        #预测
        predict_dataset = torch.from_numpy(test1)
       
        model.eval()#测试的时候加载模型之后要指定eval模式，迁移学习不用，还是modle.train（）

        """使用PyTorch进行训练和测试时一定注意要把实例化的model指定train/eval，
        eval（）时，框架会自动把BN和DropOut固定住，不会取平均，而是用训练好的值"""

        #with torch.no_grad():
        #不需要保存梯度反向传播更新参数，可以减少显存使用，with torch.no_grad是不保存所有的梯度
        #torch.squeeze维度压缩，不指定dim就删除所有大小为1的维
        print('output shape:')
        output = torch.squeeze(model(predict_dataset.to(device))).cpu()
        result = output.detach().numpy()
        print('result min and max:',result.min(),result.max())

        obs = testlable.variables['pr'][:, tg_mn, 0, 0]
        
        obs0 = torch.from_numpy(obs)
        loss_function = torch.nn.MSELoss()
        loss = loss_function(output, obs0)
        
        print('test loss:',loss)
        testloss[epoch] = loss
        
        result =result / np.std(result)
        obs = obs / np.std(obs)
        cor[epoch] = np.round(np.corrcoef(obs[3:], result[3:])[0, 1], 5)
        print('epoch,cor:',epoch,cor[epoch])
       

        #画热图,2代求一次
        if epoch % 2 == 0:
            transinput = torch.tensor(inpv1,requires_grad=True)
            print('outputtrans shape')
            outputtrans = model(transinput.to(device))
            outputtrans.backward(torch.ones_like(outputtrans))
            print('gradtrans shape:',transinput.grad.shape)
            gradtrans = transinput.grad.numpy()
            gradtransabs = abs(gradtrans)
            print('gradtransabs shape:',gradtransabs.shape)
                
            a = epoch // 2 #取整
            grad[a] = np.mean(gradtransabs,axis=0)

    
    grad.astype('float32').tofile('/content/drive/MyDrive/ENnumber/transgrad.gdat')
    testloss.astype('float32').tofile('/content/drive/MyDrive/ENnumber/testloss.gdat')
    cor.astype('float32').tofile('/content/drive/MyDrive/ENnumber/testcor.gdat')
    
if __name__ == "__main__":
    main()    