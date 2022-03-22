import torch
from .base_model import BaseModel
from . import networks
import torch.nn as nn
from .networks import init_net

class linear(nn.Module):
    def __init__(self, indim, outdim):
        super(linear, self).__init__()
        self.add_module('linearMap', nn.Linear(indim, outdim))
    def forward(self, x):
        return self.linearMap(x)


class vaeModel(BaseModel):
    """ This class implements the pix2pix model, for learning a mapping from input images to output images given paired data.

    The model training requires '--dataset_mode aligned' dataset.
    By default, it uses a '--netG unet256' U-Net generator,
    a '--netD basic' discriminator (PatchGAN),
    and a '--gan_mode' vanilla GAN loss (the cross-entropy objective used in the orignal GAN paper).

    pix2pix paper: https://arxiv.org/pdf/1611.07004.pdf
    """


    def __init__(self, opt):
        """Initialize the pix2pix class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseModel.__init__(self, opt)
        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ['kl', 'G_L1']
        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
        self.visual_names = ['real_A', 'fake_B', 'real_B']
        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>
        # if self.isTrain:
        #     self.model_names = ['G', 'D']
        # else:  # during test time, only load G
        #     self.model_names = ['G']
        self.model_names = ['_encoder', '_to_mu', '_to_var', '_upscale','G']

        # define networks (both generator and discriminator)
        self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm,
                                      not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)

        self.net_encoder = networks.define_D(opt.input_nc, opt.ndf, opt.netD,
                                          opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)
                            # networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm,
                            #           not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)

        self.net_to_mu = init_net(linear(30*30, 128), init_type= opt.init_type, init_gain=opt.init_gain, gpu_ids=self.gpu_ids)#nn.Linear(256*256, 128).to(self.gpu_ids[0])
        self.net_to_var = init_net(linear(30*30, 128), init_type= opt.init_type, init_gain=opt.init_gain, gpu_ids=self.gpu_ids) # nn.Linear(256*256, 128).to(self.gpu_ids[0])
        self.net_upscale = init_net(linear(128, 256*256), init_type= opt.init_type, init_gain=opt.init_gain, gpu_ids=self.gpu_ids)#nn.Linear(128, 256*256).to(self.gpu_ids[0])


        print(self.net_to_mu.module)
        # if self.isTrain:  # define a discriminator; conditional GANs need to take both input and output images; Therefore, #channels for D is input_nc + output_nc
        #     self.netD = networks.define_D(opt.input_nc + opt.output_nc, opt.ndf, opt.netD,
        #                                   opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)

        # if self.isTrain:
        #     # define loss functions
        #     self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)
        #     self.criterionL1 = torch.nn.L1Loss()
        #     # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
        #
        #     self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
        #     self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
        #     self.optimizers.append(self.optimizer_G)
        #     self.optimizers.append(self.optimizer_D)

        self.criterionL1 = torch.nn.L1Loss()

        self.optimizer = torch.optim.Adam([{'params': self.net_encoder.parameters()},
                                           {'params': self.net_to_mu.parameters()},
                                           {'params': self.net_to_var.parameters()},
                                           {'params': self.net_upscale.parameters()},
                                           {'params': self.netG.parameters()}
                                           ], lr = opt.lr, betas=(opt.beta1, 0.999))
        self.optimizers.append(self.optimizer)
        self.opt = opt

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): include the data itself and its metadata information.

        The option 'direction' can be used to swap images in domain A and domain B.
        """
        AtoB = self.opt.direction == 'AtoB'
        self.real_A = input['A' if AtoB else 'B'].to(self.device)
        self.real_B = input['B' if AtoB else 'A'].to(self.device)
        self.image_paths = input['A_paths' if AtoB else 'B_paths']

    def reparameter(self, mu, var):
        var = torch.exp(0.5*var)
        eps = torch.randn_like(mu)
        z = mu + eps * var
        return z

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""

        # self.fake_B = self.netG(self.real_A)  # G(A)
        # print('begin forward')
        h = self.net_encoder(self.real_B)
        h_flatten = h.view(h.shape[0], -1)
        self.mu = self.net_to_mu(h_flatten)
        self.var = self.net_to_var(h_flatten)
        h_reparam = self.reparameter(self.mu, self.var)
        h_upscale = self.net_upscale(h_reparam)
        h_decoder_input = h_upscale.view(h.shape[0], 1, 256, 256)
        self.fake_B = self.netG(h_decoder_input)

    def backward(self):

        self.loss_kl = (-0.5 * torch.sum(1 + self.var - self.mu.pow(2) - self.var.exp()) ) * 0.0001
        self.loss_G_L1 = self.criterionL1(self.fake_B, self.real_B) * 10
        self.loss_G = self.loss_G_L1 + self.loss_kl

        self.loss_G.backward()

    # def backward_D(self):
    #     """Calculate GAN loss for the discriminator"""
    #     # Fake; stop backprop to the generator by detaching fake_B
    #     fake_AB = torch.cat((self.real_A, self.fake_B), 1)  # we use conditional GANs; we need to feed both input and output to the discriminator
    #     pred_fake = self.netD(fake_AB.detach())
    #     self.loss_D_fake = self.criterionGAN(pred_fake, False)
    #     # Real
    #     real_AB = torch.cat((self.real_A, self.real_B), 1)
    #     pred_real = self.netD(real_AB)
    #     self.loss_D_real = self.criterionGAN(pred_real, True)
    #     # combine loss and calculate gradients
    #     self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5
    #     self.loss_D.backward()
    #
    # def backward_G(self):
    #     """Calculate GAN and L1 loss for the generator"""
    #     # First, G(A) should fake the discriminator
    #     fake_AB = torch.cat((self.real_A, self.fake_B), 1)
    #     pred_fake = self.netD(fake_AB)
    #     self.loss_G_GAN = self.criterionGAN(pred_fake, True)
    #     # Second, G(A) = B
    #     self.loss_G_L1 = self.criterionL1(self.fake_B, self.real_B) * self.opt.lambda_L1
    #
    #     # self.loss_G_reg = self.fake_B.sum()
    #     # g_reg_weight = 3e-6
    #
    #     # combine loss and calculate gradients
    #     self.loss_G = self.loss_G_GAN + self.loss_G_L1
    #
    #
    #
    #     self.loss_G.backward()

    def optimize_parameters(self):
        self.forward()                   # compute fake images: G(A)
        # update D
        # self.set_requires_grad(self.netD, True)  # enable backprop for D
        # self.optimizer_D.zero_grad()     # set D's gradients to zero
        # self.backward_D()                # calculate gradients for D
        # self.optimizer_D.step()          # update D's weights
        #
        # # weight clipping of WGAN
        # if (self.opt.gan_mode == 'wgangp'):
        #     for p in self.netD.parameters():
        #         p.data.clamp_(-self.opt.clip_value, self.opt.clip_value)

        # update G
        # self.set_requires_grad(self.netD, False)  # D requires no gradients when optimizing G
        self.optimizer.zero_grad()        # set G's gradients to zero
        self.backward()                   # calculate graidents for G
        self.optimizer.step()             # udpate G's weights


