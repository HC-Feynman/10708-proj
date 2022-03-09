import torch
import torch.nn as nn
import os
from PIL import Image
import numpy as np

# Modify from https://github.com/ZongxianLee/MMD_Loss.Pytorch
class MMD_loss(nn.Module):
    def __init__(self, kernel_mul = 2.0, kernel_num = 5):
        super(MMD_loss, self).__init__()
        self.kernel_num = kernel_num
        self.kernel_mul = kernel_mul
        self.fix_sigma = None

    def guassian_kernel(self, source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
        n_samples = int(source.size()[0])+int(target.size()[0])

        total = torch.cat([source, target], dim=0)

        total0 = total.unsqueeze(0).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
        total1 = total.unsqueeze(1).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
        L2_distance = ((total0-total1)**2).sum(2)
        if fix_sigma:
            bandwidth = fix_sigma
        else:
            bandwidth = torch.sum(L2_distance.data) / (n_samples**2-n_samples)
        bandwidth /= kernel_mul ** (kernel_num // 2)
        bandwidth_list = [bandwidth * (kernel_mul**i) for i in range(kernel_num)]
        kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]
        return sum(kernel_val)

    def forward(self, source, target):
        batch_size = int(source.size()[0])
        kernels = self.guassian_kernel(source, target, kernel_mul=self.kernel_mul, kernel_num=self.kernel_num, fix_sigma=self.fix_sigma)
        XX = kernels[:batch_size, :batch_size]
        YY = kernels[batch_size:, batch_size:]
        XY = kernels[:batch_size, batch_size:]
        YX = kernels[batch_size:, :batch_size]
        loss = torch.mean(XX + YY - XY -YX)
        return loss

def binary_data(inp, maxval = 255.0):

    return (inp > 0.5 * maxval) * 1.

def error(path_to_real, path_to_generate, binary, type):

    generate_files = os.listdir(path_to_generate)
    real_files = os.listdir(path_to_real)

    real_files = [t for t in real_files if not t[0] == '.']

    assert len(real_files) == len(generate_files)

    real_list = []
    generate_list = []
    for i in range(len(real_files)):
        dreal = os.path.join(path_to_real, real_files[i])
        dgene = os.path.join(path_to_generate, generate_files[i])
        img_real = Image.open(dreal).convert('L')
        img_generate = Image.open(dgene).convert('L')
        img_real = np.expand_dims(np.asarray(img_real),axis=0)
        img_generate = np.expand_dims(np.asarray(img_generate),axis=0)

        if binary == True:
            img_real = binary_data(img_real)
            img_generate = binary_data(img_generate)
        real_list.append(img_real)
        generate_list.append(img_generate)
    img_real = np.concatenate(real_list).reshape(len(real_files), -1)
    img_generate = np.concatenate(generate_list).reshape(len(real_files), -1)

    if type == 'MMD':
        mmd = MMD_loss()
        img_real = torch.Tensor(img_real)
        img_generate = torch.Tensor(img_generate)
        loss = mmd.forward(img_real, img_generate)
    elif type == 'L1':
        loss = 0
        for i in range(len(real_list)):
            loss += np.linalg.norm(real_list[i].reshape(-1) - generate_list[i].reshape(-1), ord=1, keepdims=True)
        loss = loss/len(real_list)

    print(loss)


path_to_real = '/Users/huichen/Desktop/10708/Project/实验结果/一对多downsample baseline Feb 17/test_latest/images/for-metric/real'
path_to_generate = '/Users/huichen/Desktop/10708/Project/实验结果/一对多downsample baseline Feb 17/test_latest/images/for-metric/generated'
binary = False
type = "L1"#'MMD' # or 'L1'
error(path_to_real, path_to_generate, binary, type)
