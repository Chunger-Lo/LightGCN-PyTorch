import torch
import numpy as np
from os.path import join
from tensorboardX import SummaryWriter
import time
from datetime import timedelta
import utils
import world
from world import cprint, dataset
from model import LightGCN
import Procedure
import dataloader

# ==============================
utils.set_seed(world.seed)
print(">>SEED:", world.seed)
# ==============================
# import register
# from register import dataset

# revise
dataset = dataloader.Loader(path="../data/"+world.dataset)

start = time.time()
Recmodel = LightGCN(world.config, dataset)
Recmodel = Recmodel.to(world.device)
bpr = utils.BPRLoss(Recmodel, world.config)

weight_filename = utils.getFileName()
cprint(f"load or save to {weight_filename}")

if world.mode == 'fastdebug':
    
    for _bpr_size in world.config['bpr_batch_size']:
        try:
        # init tensorboard
            if world.tensorboard:
                date = time.strftime("%m%d")[-4:]
                te_date = world.config["test_date"][-4:]
                tensor_log_path = join(world.BOARD_PATH, f'{date}-te{te_date}_fastdebug')
                w : SummaryWriter = SummaryWriter(tensor_log_path)
                cprint(f"Write to: {tensor_log_path}")
            else:
                w = None
                cprint("not enable tensorflowboard")
            for epoch in [1]:
                start = time.time()
                output_information = Procedure.BPR_train_original(dataset, Recmodel, bpr, _bpr_size, epoch, neg_k= world.config['negK'],w=w)
                print(f'EPOCH[1] {output_information}')
                if True:
                    cprint("[Validation]")
                    Procedure.Test(dataset, Recmodel, w, multicore = world.config['multicore'])
            print(f'state_dict:  {Recmodel.state_dict()}')
            torch.save(Recmodel.state_dict(), weight_filename)
            end = time.time()
            time_elasped = end - start

            print(f'Time elasped: {str(timedelta(minutes=end - start))}')
        finally:
            if world.tensorboard:
                w.close()
if world.mode == 'train':
    for _bpr_size in world.config['bpr_batch_size']:
        try:
        # init tensorboard
            if world.tensorboard:
                date = time.strftime("%m%d")[-4:]
                te_date = world.config["test_date"][-4:]
                my_comment = f'{date}-te{te_date}_{world.comment}_bpr{str(_bpr_size)}'
                tensor_log_path = join(world.BOARD_PATH, my_comment)
                w : SummaryWriter = SummaryWriter(tensor_log_path)
                world.cprint(f"Write to {tensor_log_path}")
            else:
                w = None
                world.cprint("not enable tensorflowboard")
            for epoch in range(world.TRAIN_epochs):
                start = time.time()
                output_information = Procedure.BPR_train_original(dataset, Recmodel, bpr, _bpr_size, epoch, neg_k=world.config['negK'],w=w)
                print(f'EPOCH[{epoch+1}/{world.TRAIN_epochs}] {output_information}')
                # if True:
                if epoch%5 == 0:
                    cprint("[Validation]]")
                    Procedure.Test(dataset, Recmodel, epoch, w, multicore = world.config['multicore'])
            torch.save(Recmodel.state_dict(), weight_filename)
            end = time.time()
            time_elasped = end - start

            print(f'Time elasped: {str(timedelta(minutes=end - start))}')
        finally:
            if world.tensorboard:
                w.close()
if world.mode == 'test':
    world.cprint("Starting test mode")
    try:
    # init tensorboard
        if world.tensorboard:
            date = time.strftime("%m%d")[-4:]
            te_date = world.config["test_date"][-4:]
            bsize = world.config['test_u_batch_size']
            my_comment = f'{date}-te{te_date}_{world.comment}_te_bsize{bsize}'
            tensor_log_path = join(world.BOARD_PATH, my_comment)
            w : SummaryWriter = SummaryWriter(tensor_log_path)
            world.cprint(f"Write to {tensor_log_path}")
        else:
            w = None
            world.cprint("not enable tensorflowboard")
        #loading
        # torch.save(Recmodel.state_dict(), weight_file)
        if world.LOAD:
            try:
                Recmodel.load_state_dict(torch.load(weight_filename,map_location=torch.device('cpu')))
                world.cprint(f"loaded model weights from {weight_filename}")
            except FileNotFoundError:
                print(f"{weight_filename} not exists, start from beginning")

        cprint("[TEST]")
        start = time.time()
        Procedure.Test(dataset, Recmodel, w, multicore = world.config['multicore'], test_type = 'all')
        Procedure.Test(dataset, Recmodel, w, multicore = world.config['multicore'], test_type = 'pos_only')

        end = time.time()
        time_elasped = end - start

        print(f'Time elasped: {str(timedelta(minutes=end - start))}')
    finally:
        if world.tensorboard:
            w.close()

end = time.time()
time_elasped = end - start

print(f'Time elasped: {str(timedelta(minutes=end - start))}')