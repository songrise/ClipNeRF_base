import torch
from torch import nn
from criteria.clip_loss import CLIPLoss
class MSELoss(nn.Module):
    def __init__(self):
        super(MSELoss, self).__init__()
        self.loss = nn.MSELoss(reduction='mean')

    def forward(self, inputs, targets):
        #! Jun 18: the "inputs" is the dict of results from the model
        loss = self.loss(inputs['rgb_coarse'], targets)
        if 'rgb_fine' in inputs:
            loss += self.loss(inputs['rgb_fine'], targets)

        return loss
               
class DirClipLoss(nn.Module):
    """wrapper for clip loss"""
    def __init__(self, device):
        super(DirClipLoss, self).__init__()
        self.loss = CLIPLoss(device)


    def forward(self, src_img, src_text, results, target_text, H = 60, W = 60):
        #! Jun 18: the "inputs" is the dict of results from the model
        H, W = int(H), int(W)

        target_img = results['rgb_fine']
        target_img = self.cvt_CHW(target_img, H, W)
        src_img = self.cvt_CHW(src_img, H, W)

        loss = self.loss.clip_directional_loss(src_img, src_text, target_img, target_text)
        if 'rgb_coarse' in results:
            target_img_coarse = results['rgb_coarse']
            target_img_coarse = self.cvt_CHW(target_img_coarse, H, W)

            loss += self.loss.clip_directional_loss(src_img, src_text, target_img_coarse, target_text)
        return loss
    
    def cvt_CHW(self, img:torch.Tensor, H, W):
        """
        convert (1, H, W, 3) to (1, 3, H, W)
        """
        img = img.view(1, H, W, 3) 
        img = img.permute(0, 3, 1, 2) # (1, 3, H, W) 
        return img

loss_dict = {'mse': MSELoss,
            'dirClip':DirClipLoss}

if __name__ == "__main__":
    device = torch.device('cuda:0')
    dirLoss = DirClipLoss(device)
    # for _ in dirLoss.parameters():
    #     print(_)