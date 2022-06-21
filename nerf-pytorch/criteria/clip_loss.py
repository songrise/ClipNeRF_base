# -*- coding : utf-8 -*-
# @FileName  : clip_loss.py
# @Author    : Ruixiang JIANG (Songrise)
# @Time      : Jun 11, 2022
# @Github    : https://github.com/songrise
# @Description: Temporarily use this naive clip loss

import torch
import clip


class CLIPLoss(torch.nn.Module):

    def __init__(self):
        super(CLIPLoss, self).__init__()
        self.model, self.preprocess = clip.load("ViT-B/32", device="cuda")


    def forward(self, image, text):
        image = torch.nn.functional.upsample_bilinear(image, (224, 224))
        # image = self.avg_pool(self.upsample(image))
        similarity = 1 - self.model(image, text)[0] / 100
        return similarity