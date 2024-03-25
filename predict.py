import torch
import numpy as np
from netCDF4 import Dataset

from model import CNN

devices = 'cuda:0'
num_conv = 3
num_hidd = 3

lead_mon = 6
target_mon = 1
tg_mn = int(target_mon - 1)
ld_mn1 = int(23 - lead_mon + tg_mn)
ld_mn2 = int(23 - lead_mon + tg_mn + 3)

'''测试集'''
test = Dataset('/content/drive/MyDrive/input/GODAS.input.36mon.1980_2015.nc','r')

test1 = np.zeros((36,6,24,72),dtype=np.float32)
test1[:,0:3,:,:] = test.variables['sst'][0:36,ld_mn1:ld_mn2,:,:]
test1[:,3:6,:,:] = test.variables['t300'][0:36,ld_mn1:ld_mn2,:,:]

testlable = Dataset('/content/drive/MyDrive/input/GODAS.label.12mon.1982_2017.nc', 'r')

def main():

    device = torch.device(devices if torch.cuda.is_available() else "cpu")
    print(f"using {device} device.")

    #best cmip model
    for num in range(10):
        model = CNN(num_conv=num_conv, num_hidd=num_hidd).to(device)
        weightfile ='/content/drive/MyDrive/cmip/bestcmip_model.pth'
        weights_dict = torch.load(weightfile, map_location=device)#["model"]

        print(model.load_state_dict(weights_dict, strict=True))

        #预测
        predict_dataset = torch.from_numpy(test1)

        model.eval()#测试的时候加载模型之后要指定eval模式，迁移学习不用，还是modle.train（）

        with torch.no_grad():
            #不需要保存梯度反向传播更新参数，可以减少显存使用，with torch.no_grad是不保存所有的梯度
            #torch.squeeze维度压缩，不指定dim就删除所有大小为1的维

            output = torch.squeeze(model(predict_dataset.to(device))).cpu()
            result = output.detach().numpy()
            print('result min and max:',result.min(),result.max())

        result.astype('float32').tofile('/content/drive/MyDrive/ENnumber/bmresult.gdat')
        break #只使用第一个最好的模型，所以跳出循环

if __name__ == "__main__":
    main()
