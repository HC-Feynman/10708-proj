# -*- coding: utf-8 -*-
"""

@author: Aamir Mustafa and Rafal K. Mantiuk
Implementation of the paper:
    Transformation Consistency Regularization- A Semi Supervised Paradigm for Image to Image Translation
    ECCV 2020

This file generated the Transformation Matrix that is being used to train TCR.

"""

# pip install kornia

import torch
import numpy as np
import kornia
import torch.nn as nn


class TCR(nn.Module):
    def __init__(self):
        super(TCR, self).__init__()

        angp = 30
        tp = 2
        zp = 0.5

        self.ang = np.deg2rad(angp)  # 20 # Change the degree of rotation as per the task in hand
        self.ang_neg = -1 * self.ang
        self.max_tx, self.max_ty = tp, tp  # 6.0, 6.0  # Change as per the task
        self.min_tx, self.min_ty =  -tp, -tp # -6.0, -6.0  # Change as per the task

        self.max_z, self.min_z = zp, zp # 1.00, 1.00  # Change as per the task

    #

    def forward(self, img, random):
        #        print('img.shape is', img.shape)
        bs = img.shape[0]
        #        print('bs is', bs)
        W = img.shape[2]
        H = img.shape[3]

        tx = ((self.max_tx - self.min_tx) * random + self.min_tx)
        ty = ((self.max_ty - self.min_ty) * random + self.min_ty)

        r = ((self.ang - self.ang_neg) * random + self.ang_neg)
        z = ((self.max_z - self.min_z) * random + self.min_z)

        hx = ((self.ang - self.ang_neg) * random + self.ang_neg)
        hy = ((self.ang - self.ang_neg) * random + self.ang_neg)

        if torch.cuda.is_available():
            tx = tx.to("cuda")
            ty = ty.to("cuda")
            r = r.to("cuda")
            z = z.to("cuda")
            hx = hx.to("cuda")
            hy = hy.to("cuda")

        # Transformation Matrix

        a = hx - r
        b = hy + r

        T11 = torch.div(z * torch.cos(a), torch.cos(hx))

        T12 = torch.div(z * torch.sin(a), torch.cos(hx))

        T13 = torch.div(W * torch.cos(hx) - W * z * torch.cos(a) + 2 * tx * z * torch.cos(a) - H * z * torch.sin(
            a) + 2 * ty * z * torch.sin(a), 2 * torch.cos(hx))

        T21 = torch.div(z * torch.sin(b), torch.cos(hy))

        T22 = torch.div(z * torch.cos(b), torch.cos(hy))

        T23 = torch.div(H * torch.cos(hy) - W * z * torch.cos(b) + 2 * ty * z * torch.cos(b) - W * z * torch.sin(
            b) + 2 * tx * z * torch.sin(b), 2 * torch.cos(hy))

        T = torch.zeros((bs, 2, 3))  # Combined for batch

        if torch.cuda.is_available():
            T = T.to('cuda')

        for i in range(bs):
            T[i] = torch.tensor(
                [[T11[i], T12[i], T13[i]], [T21[i], T22[i], T23[i]]])  # Transformation Matrix for a batch

        Transformed_img = kornia.geometry.transform.warp_affine(img, T, dsize=(W, H))
        if torch.cuda.is_available():
            Transformed_img = Transformed_img.to('cuda')

        return Transformed_img


