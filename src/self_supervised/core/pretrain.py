'''Author- Prakash Chandra Chhipa, Email- prakash.chandra.chhipa@ltu.se, Year- 2022'''

import os,sys
import logging
from os import get_exec_path
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
from tqdm import tqdm

import torch
from self_supervised.core import ssl_loss
from self_supervised.apply import config
sys.path.append(os.path.dirname(__file__))


def pretrain_epoch_MPCS(gpu, current_epoch, epochs, batch_size, train_loader,
                          model, optimizer, criterion):

    model.train()
    total_loss = 0
    epoch_response_dir = {}
    with tqdm(total=batch_size * len(train_loader),
              desc=f'Epoch {current_epoch}/{epochs}',
              unit='img') as (pbar):

        for idx, batch in enumerate(train_loader):
            view1, view2 = batch[0], batch[1]
            b, c, h, w = view1.size()
            #for pytorch tranform
            view1 = view1.cuda(gpu, non_blocking=True)
            view2 = view2.cuda(gpu, non_blocking=True)

            output_view1 = model(view1)
            output_view2 = model(view2)

            output = torch.cat(
                [output_view1.unsqueeze(1),
                 output_view2.unsqueeze(1)], dim=1)
            loss = criterion(output)
            curr_loss = loss.item()
            total_loss += curr_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            '''logging'''
            #logging.info('minibatch: {idx} simCLR running_loss: {loss.item()}')
            (pbar.set_postfix)(**{'loss (batch)': loss.item()})
            pbar.update(view1.shape[0])
            

        # Prepare epoch reponse and return
        epoch_response_dir['model'] = model
        epoch_response_dir['loss'] = total_loss/(batch_size*len(train_loader))
        epoch_response_dir['image_pair'] = [view1, view2]

    return epoch_response_dir