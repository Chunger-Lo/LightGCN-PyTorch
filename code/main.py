import world
import utils
from world import cprint
import torch
import numpy as np
from tensorboardX import SummaryWriter
import time
import Procedure
from os.path import join
# ==============================
utils.set_seed(world.seed)
print(">>SEED:", world.seed)
# ==============================
import register
from register import dataset
import time
from datetime import timedelta


start = time.time()

Recmodel = register.MODELS[world.model_name](world.config, dataset)
Recmodel = Recmodel.to(world.device)
bpr = utils.BPRLoss(Recmodel, world.config)

weight_file = utils.getFileName()
print(f"load and save to {weight_file}")
if world.LOAD:
    try:
        Recmodel.load_state_dict(torch.load(weight_file,map_location=torch.device('cpu')))
        world.cprint(f"loaded model weights from {weight_file}")
    except FileNotFoundError:
        print(f"{weight_file} not exists, start from beginning")
Neg_k = 1

if world.mode == 'fastdebug':
    
    for _bpr_size in world.config['bpr_batch_size']:
        try:
        # init tensorboard
            if world.tensorboard:
                tensor_log_path = join(world.BOARD_PATH, 
                                                time.strftime("%m%d") + "-" + "te" + world.config["test_date"]+"_" + world.comment+\
                                                    '_' + 'bpr' +str(_bpr_size))
                w : SummaryWriter = SummaryWriter(tensor_log_path)
                world.cprint(f"Write to {tensor_log_path}")
            else:
                w = None
                world.cprint("not enable tensorflowboard")
            for epoch in [1]:
                start = time.time()
                output_information = Procedure.BPR_train_original(dataset, Recmodel, bpr, _bpr_size, epoch, neg_k=Neg_k,w=w)
                print(f'EPOCH[1] {output_information}')
                if True:
                    cprint("[TEST]")
                    Procedure.Test(dataset, Recmodel, epoch, w, multicore = world.config['multicore'])
            torch.save(Recmodel.state_dict(), weight_file)
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
                date = time.strftime("%m%d")
                my_comment = date + "-" + "te" + world.config["test_date"]+"_" + world.comment+\
                            '_' + 'bpr' +str(_bpr_size)
                MyComment = f'{date}_{world.config["test_date"]}_{world.comment}_bpr'
                tensor_log_path = join(world.BOARD_PATH, my_comment)
                w : SummaryWriter = SummaryWriter(tensor_log_path)
                world.cprint(f"Write to {tensor_log_path}")
            else:
                w = None
                world.cprint("not enable tensorflowboard")
            for epoch in range(world.TRAIN_epochs):
                start = time.time()
                output_information = Procedure.BPR_train_original(dataset, Recmodel, bpr, _bpr_size, epoch, neg_k=Neg_k,w=w)
                print(f'EPOCH[{epoch+1}/{world.TRAIN_epochs}] {output_information}')
                if True:
                    cprint("[TEST]")
                    Procedure.Test(dataset, Recmodel, epoch, w, multicore = world.config['multicore'])
            torch.save(Recmodel.state_dict(), weight_file)
            end = time.time()
            time_elasped = end - start

            print(f'Time elasped: {str(timedelta(minutes=end - start))}')
        finally:
            if world.tensorboard:
                w.close()
# if world.mode == 'train':
    # try:
    #     for epoch in range(world.TRAIN_epochs):
    #         start = time.time()
    #         output_information = Procedure.BPR_train_original(dataset, Recmodel, bpr, epoch, neg_k=Neg_k,w=w)
    #         print(f'EPOCH[{epoch+1}/{world.TRAIN_epochs}] {output_information}')
    #         if epoch %10 == 0:
    #             cprint("[TEST]")
    #             Procedure.Test(dataset, Recmodel, epoch, w, multicore = world.config['multicore'])
    #         torch.save(Recmodel.state_dict(), weight_file)
    # finally:
    #     if world.tensorboard:
    #         w.close()
# try:
#     for epoch in range(world.TRAIN_epochs):
#         start = time.time()
#         output_information = Procedure.BPR_train_original(dataset, Recmodel, bpr, epoch, neg_k=Neg_k,w=w)
#         print(f'EPOCH[{epoch+1}/{world.TRAIN_epochs}] {output_information}')
#         if (epoch+1) %10 == 0:
#         # if True:
#             cprint("[TEST]")
#             Procedure.Test(dataset, Recmodel, epoch, w, multicore = world.config['multicore'])
#         torch.save(Recmodel.state_dict(), weight_file)
# try:
#     for _bpr_size in world.config['bpr_batch_size']:
#         for epoch in range(world.TRAIN_epochs):
#             start = time.time()
#             output_information = Procedure.BPR_train_original(dataset, Recmodel, bpr, _bpr_size, epoch, neg_k=Neg_k,w=w)
#             print(f'EPOCH[{epoch+1}/{world.TRAIN_epochs}] {output_information}')
#             if (epoch+1) %10 == 0:
#             # if True:
#                 cprint("[TEST]")
#                 Procedure.Test(dataset, Recmodel, epoch, w, multicore = world.config['multicore'])
#             torch.save(Recmodel.state_dict(), weight_file)
# finally:
#     if world.tensorboard:
#         w.close()

end = time.time()
time_elasped = end - start

print(f'Time elasped: {str(timedelta(minutes=end - start))}')