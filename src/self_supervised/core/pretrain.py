'''Author- Prakash Chandra Chhipa, Email- prakash.chandra.chhipa@ltu.se/prakash.chandra.chhipa@gmail.com, Year- 2022'''

import os,sys, math, json, time
import logging
from os import get_exec_path
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
from tqdm import tqdm

import torch
from self_supervised.core import ssl_loss
from self_supervised.apply import config
sys.path.append(os.path.dirname(__file__))

def adjust_learning_rate(epochs, batch_size, learning_rate_weights, learning_rate_biases, optimizer, train_loader, step):
    max_steps = epochs * len(train_loader)
    warmup_steps = 10 * len(train_loader)
    base_lr = batch_size / 256
    if step < warmup_steps:
        lr = base_lr * step / warmup_steps
    else:
        step -= warmup_steps
        max_steps -= warmup_steps
        q = 0.5 * (1 + math.cos(math.pi * step / max_steps))
        end_lr = base_lr * 0.001
        lr = base_lr * q + end_lr * (1 - q)
    optimizer.param_groups[0]['lr'] = lr * learning_rate_weights
    optimizer.param_groups[1]['lr'] = lr * learning_rate_biases

def pretrain_epoch_MPCS_multi_gpu(gpu, current_epoch, epochs, batch_size, train_loader,
                          model, optimizer, criterion, scaler, learning_rate_weights,
        learning_rate_biases,
        print_freq,
        stats_file,
        start_time,
        rank,
        optimizer_type,
        single_gpu=False):

    model.train()
    total_loss = 0
    epoch_response_dir = {}
    with tqdm(total=batch_size * len(train_loader),
              desc=f'Epoch {current_epoch}/{epochs}',
              unit='img') as (pbar):

        for step, batch in enumerate(train_loader):
            view1, view2 = batch[0], batch[1]
        
            #for pytorch tranform
            view1 = view1.cuda(gpu, non_blocking=True)
            view2 = view2.cuda(gpu, non_blocking=True)
            if True == optimizer_type:
                adjust_learning_rate(epochs, batch_size, learning_rate_weights,
                                 learning_rate_biases, optimizer, train_loader, step)
            optimizer.zero_grad()
            if single_gpu == False:
                with torch.cuda.amp.autocast():
                    output_view1 = model(view1)
                    output_view2 = model(view2)
                    loss = criterion(output_view1, output_view2)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                total_loss += loss.item()   
            else:
                output_view1 = model(view1)
                output_view2 = model(view2)
                loss = criterion(output_view1, output_view2)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()  
            if step % print_freq == 0:
                if rank == 0:
                    stats = dict(epoch=current_epoch, step=step,
                                 lr_weights=optimizer.param_groups[0]['lr'],
                                 lr_biases=optimizer.param_groups[1]['lr'],
                                 loss=loss.item(),
                                 time=int(time.time() - start_time))
                    print(json.dumps(stats))
                    print(json.dumps(stats), file=stats_file)
            
            
            '''logging'''
            (pbar.set_postfix)(**{'loss (batch)': loss.item()})
            pbar.update(view1.shape[0])

        state = dict(epoch=current_epoch + 1, model=model.state_dict(),
                     optimizer=optimizer.state_dict())
        # Prepare epoch reponse and return
        epoch_response_dir['state'] = state
        epoch_response_dir['model'] = model
        if False == single_gpu:
            epoch_response_dir['backbone'] = model.module.backbone.state_dict()
        else:
            epoch_response_dir['backbone'] = model.module.backbone.state_dict()
        epoch_response_dir['loss'] = total_loss/(batch_size*len(train_loader))
        epoch_response_dir['image_pair'] = [view1, view2]

    return epoch_response_dir



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
            (pbar.set_postfix)(**{'loss (batch)': loss.item()})
            pbar.update(view1.shape[0])
            

        # Prepare epoch reponse and return
        epoch_response_dir['model'] = model
        epoch_response_dir['loss'] = total_loss/(batch_size*len(train_loader))
        epoch_response_dir['image_pair'] = [view1, view2]

    return epoch_response_dir