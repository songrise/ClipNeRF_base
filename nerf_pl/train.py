import os, sys
from opt import get_opts
import torch
from collections import defaultdict

from torch.utils.data import DataLoader
from datasets import dataset_dict

# models
from models.nerf import Embedding, NeRF
from models.rendering import render_rays

# data
from datasets.rand_batch import RandBatchSampler

# optimizer, scheduler, visualization
from utils import *

# losses
from losses import loss_dict
from losses import DirClipLoss

# metrics
from metrics import *

# pytorch-lightning
from pytorch_lightning.callbacks import ModelCheckpoint, Callback
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.logging import TestTubeLogger


class NeRFSystem(LightningModule):
    def __init__(self, hparams):
        super(NeRFSystem, self).__init__()
        self.hparams = hparams
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        if hparams.loss_type == 'mse':
            self.loss = loss_dict[hparams.loss_type]()
        elif hparams.loss_type == 'dirClip':
            #! Jun 20: else, init loss laters
            self.loss = None

        self.embedding_xyz = Embedding(3, 10) # 10 is the default number
        self.embedding_dir = Embedding(3, 4) # 4 is the default number
        self.embeddings = [self.embedding_xyz, self.embedding_dir]

        self.nerf_coarse = NeRF()
        self.models = [self.nerf_coarse]
        if hparams.N_importance > 0:
            self.nerf_fine = NeRF()
            self.models += [self.nerf_fine]

    def decode_batch(self, batch):
        #! Jun 18: r_0, r_d, near, far
        rays = batch['rays'] # (B, 8) 
        rgbs = batch['rgbs'] # (B, 3)
        return rays, rgbs

    def forward(self, rays):
        """Do batched inference on rays using chunk."""
        B = rays.shape[0]
        results = defaultdict(list)
        for i in range(0, B, self.hparams.chunk):
            rendered_ray_chunks = \
                render_rays(self.models,
                            self.embeddings,
                            rays[i:i+self.hparams.chunk],
                            self.hparams.N_samples,
                            self.hparams.use_disp,
                            self.hparams.perturb,
                            self.hparams.noise_std,
                            self.hparams.N_importance,
                            self.hparams.chunk, # chunk size is effective in val mode
                            self.train_dataset.white_back)

            for k, v in rendered_ray_chunks.items():
                results[k] += [v]

        for k, v in results.items():
            results[k] = torch.cat(v, 0)
        return results

    def prepare_data(self):
        dataset = dataset_dict[self.hparams.dataset_name]
        #! Jun 22:  assume using the stride sampled dataset
        kwargs = {'root_dir': self.hparams.root_dir,
                  'img_wh': tuple(self.hparams.img_wh),
                  'stride': self.hparams.stride,
                  'patch_size': self.hparams.patch_size}
        if self.hparams.dataset_name == 'llff':
            kwargs['spheric_poses'] = self.hparams.spheric_poses
            kwargs['val_num'] = self.hparams.num_gpus
            
        self.train_dataset = dataset(split='train', **kwargs)
        self.val_dataset = dataset(split='val', **kwargs)
        



    def configure_optimizers(self):
        self.optimizer = get_optimizer(self.hparams, self.models)
        if self.hparams.lr_scheduler == 'none':
            return self.optimizer

        scheduler = get_scheduler(self.hparams, self.optimizer)
        
        return [self.optimizer], [scheduler]
        # return [self.optimizer]


    def train_dataloader(self):
        #! Jun 19: permute patches
        train_idx = torch.arange(len(self.train_dataset))
        sampler = RandBatchSampler(train_idx, self.hparams.patch_size)
        self.patch_w = int(np.sqrt(self.hparams.patch_size))# assume all patches are square
        #! Jun 22: shuffle disabled for finetuning
        return DataLoader(self.train_dataset,
                          num_workers=4,
                          shuffle=False,
                          batch_size=self.hparams.patch_size, #! always infer on patch at a time
                          pin_memory=True,
                          )


    def val_dataloader(self):
        return DataLoader(self.val_dataset,
                          shuffle=False,
                          num_workers=4,
                          batch_size=1, # validate one image (H*W rays) at a time
                          pin_memory=True)
    
    def training_step(self, batch, batch_nb):
        #! Jun 18:  batch: rays [1024, 8], rgbs [1024, 3]

        log = {'lr': get_learning_rate(self.optimizer)}
        rays, rgbs = self.decode_batch(batch)
        #! Jun 18: inference here
        results = self(rays)


        if self.hparams.loss_type == 'mse':
            log['train/loss'] = loss = self.loss(results, rgbs)
        elif self.hparams.loss_type == 'dirClip':
            log['train/clip_loss'] = loss = self.loss(rgbs,self.hparams.src_text, results, \
                self.hparams.target_text, self.patch_w,self.patch_w)
        
        #! dump rgbs to test
        # import pickle
        # with open('test_rgbs.pkl', 'wb') as f:
        #     pickle.dump(rgbs, f)
        # with open('test_results.pkl', 'wb') as f:
        #     pickle.dump(results, f)

        typ = 'fine' if 'rgb_fine' in results else 'coarse'

        with torch.no_grad():
            psnr_ = psnr(results[f'rgb_{typ}'], rgbs)
            log['train/psnr'] = psnr_

        return {'loss': loss,
                'progress_bar': {'train_psnr': psnr_},
                'log': log
               }

    def validation_step(self, batch, batch_nb):
        rays, rgbs = self.decode_batch(batch)
        rays = rays.squeeze() # (H*W, 3)
        rgbs = rgbs.squeeze() # (H*W, 3)
        results = self(rays)
        
        if self.hparams.loss_type == 'dirClip':
            self.loss = DirClipLoss(self.device)
            print("Clip loss model loaded")

        if self.hparams.loss_type == 'mse':
            log = {'val_loss': self.loss(results, rgbs)}
        elif self.hparams.loss_type == 'dirClip':
            #todo check hw
            log = {'val_loss': self.loss(rgbs,self.hparams.src_text, results, self.hparams.target_text, H=self.hparams.img_wh[0], W=self.hparams.img_wh[1])}

        typ = 'fine' if 'rgb_fine' in results else 'coarse'
    
        if batch_nb == 0:
            W, H = self.hparams.img_wh
            img = results[f'rgb_{typ}'].view(H, W, 3).cpu()
            img = img.permute(2, 0, 1) # (3, H, W)
            # assume the gt and pred are always in the same scale
            img_gt = rgbs.view(H, W, 3).permute(2, 0, 1).cpu() # (3, H, W)
            depth = visualize_depth(results[f'depth_{typ}'].view(H, W)) # (3, H, W)
            stack = torch.stack([img_gt, img, depth]) # (3, 3, H, W)
            self.logger.experiment.add_images('val/GT_pred_depth',
                                               stack, self.global_step)

        log['val_psnr'] = psnr(results[f'rgb_{typ}'], rgbs)
        return log

    def validation_epoch_end(self, outputs):
        mean_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        mean_psnr = torch.stack([x['val_psnr'] for x in outputs]).mean()

        return {'progress_bar': {'val_loss': mean_loss,
                                 'val_psnr': mean_psnr},
                'log': {'val/loss': mean_loss,
                        'val/psnr': mean_psnr}
               }


if __name__ == '__main__':
    hparams = get_opts()
    system = NeRFSystem(hparams)
    checkpoint_callback = ModelCheckpoint(filepath=os.path.join(f'ckpts/{hparams.exp_name}',
                                                                '{epoch:d}'),
                                          monitor='val/loss',
                                          mode='min',
                                          save_top_k=5,)
    # checkpoint_callback = CheckpointEveryNSteps(1500)

    logger = TestTubeLogger(
        save_dir="logs",
        name=hparams.exp_name,
        debug=False,
        create_git_tag=False
    )

    trainer = Trainer(max_epochs=hparams.num_epochs,
                      checkpoint_callback=checkpoint_callback,
                      resume_from_checkpoint=hparams.ckpt_path,
                      logger=logger,
                      early_stop_callback=None,
                      weights_summary=None,
                      progress_bar_refresh_rate=1,
                      gpus=hparams.num_gpus,
                      distributed_backend='ddp' if hparams.num_gpus>1 else None,
                      num_sanity_val_steps=1,
                      benchmark=True,
                      profiler=hparams.num_gpus==1,
                      val_check_interval=0.1)

    trainer.fit(system)
