
    with torch.no_grad():
        sample = self.train_dataset[i]
        rays, rgbs, ts = sample['rays'].to(device), sample['rgbs'].to(device), sample['ts'].to(device)
        rays = rays.squeeze()
        rgbs = rgbs.squeeze()
        ts = ts.squeeze()

        rgbs = self._prepare_for_feature_loss(rgbs)

        results = self(rays, ts, self.train_dataset.white_back)
        rendered_image = results['rgb_fine']

        transmittance = results['transmittance']
        rendered_image = self._prepare_for_feature_loss(rendered_image) # 1,3,512,320

    rendered_image.requires_grad_()
    loss = 0

    if self.hparams.use_clip:
        loss_clip = self.clip_loss(rendered_image, self.text_inputs)
        loss = loss + loss_clip * self.hparams.w_clip
    if self.hparams.use_mse:
        loss_mse = self.mse_loss(rendered_image, rgbs)
        loss = loss + loss_mse * self.hparams.w_mse
    if self.hparams.use_direct_clip:
        stext = self.hparams.stext
        ttext = self.hparams.ttext
        

    loss.backward()

    gradient = rendered_image.grad.clone().detach()

    optimizer.zero_grad()
    
    for j in range(num_patches):  # iterate over patches
        sample = self.train_dataset[i*num_patches + j]
        rays, rgbs, ts = sample['rays'].to(device), sample['rgbs'].to(device), sample['ts'].to(device)
        rays = rays.squeeze()
        ts = ts.squeeze()

        rendered_image = self(rays, ts, self.train_dataset.white_back)['rgb_fine']
        rendered_image = self._prepare_for_feature_loss(rendered_image, img_wh=(scale_w, scale_h))

        r, c = j // divide_scale, j % divide_scale
        rendered_image.backward(gradient[:, :, r * scale_h: (r + 1) * scale_h, c * scale_w: (c + 1) * scale_w])
    optimizer.step()

self.log_model(prefix=hparams.prefix, suffix=hparams.suffix, step=epoch)  # log the model

# image logging
self.log_checkpoint_image('end')

        