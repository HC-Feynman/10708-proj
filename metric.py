import torch
import torch.nn as nn
import os
from PIL import Image
import numpy as np

device = 'cuda' if torch.cuda.is_available() else 'cpu'
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
        # print(total0.shape, total1.shape)
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
        # print(kernels.shape)
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

    assert len(real_files) == len(generate_files)

    real_list = []
    generate_list = []
    for i in range(len(real_files)):
        img_real = Image.open(path_to_real+real_files[i]).convert('L')
        img_generate = Image.open(path_to_generate+generate_files[i]).convert('L')
        if img_real.size == img_generate.size and img_real.size[0] == 140 and img_real.size[1] == 70:
            img_real = np.expand_dims(np.asarray(img_real)[:,:70],axis=0)
            img_generate = np.expand_dims(np.asarray(img_generate)[:, :70],axis=0)
        elif img_real.size == img_generate.size and img_real.size[0] == 256 and img_real.size[1] == 256:
            img_real = np.expand_dims(np.asarray(img_real.resize((70,70))),axis=0)
            img_generate = np.expand_dims(np.asarray(img_generate.resize((70,70))),axis=0)
        else:
            raise Exception('image size should be 140*70 or 256*256!')

        if binary == True:
            img_real = binary_data(img_real)
            img_generate = binary_data(img_generate)
        real_list.append(img_real)
        generate_list.append(img_generate)
    img_real = np.concatenate(real_list).reshape(len(real_files), -1)
    img_generate = np.concatenate(generate_list).reshape(len(real_files), -1)

    if type == 'MMD':
        mmd = MMD_loss()
        img_real = torch.Tensor(img_real).to(device)
        img_generate = torch.Tensor(img_generate).to(device)
        loss = mmd.forward(img_real, img_generate)
    elif type == 'L1':
        loss = 0
        for i in range(len(real_list)):
            loss += np.linalg.norm(real_list[i].reshape(-1) - generate_list[i].reshape(-1), ord=1)
        loss = loss/len(real_list)

    print(type, loss)


path_to_real = './results/oracle_pix2pix/test_latest/real_B/'#'../combined_1-1 (2)/combined_1-1/test/'
path_to_generate = './results/oracle_pix2pix/test_latest/fake_B/'#'../generate_rbm_W_k=10_h=5000_size=70140/'
binary = True
type = 'L1' # or 'MMD'
error(path_to_real, path_to_generate, binary, 'L1')
error(path_to_real, path_to_generate, binary, 'MMD')

