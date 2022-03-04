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

# init tensorboard
if world.tensorboard:
    tensor_log_path = join(world.BOARD_PATH, 
                                    time.strftime("%m%d") + "-" + "te" + world.config["test_date"]+"_" + world.comment)
    w : SummaryWriter = SummaryWriter(tensor_log_path)
    world.cprint(f"Write to {tensor_log_path}")
else:
    w = None
    world.cprint("not enable tensorflowboard")

# if mode == 'fastdebug':

# if mode == 'train':
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
try:
    for epoch in range(world.TRAIN_epochs):
        start = time.time()
        output_information = Procedure.BPR_train_original(dataset, Recmodel, bpr, epoch, neg_k=Neg_k,w=w)
        print(f'EPOCH[{epoch+1}/{world.TRAIN_epochs}] {output_information}')
        if (epoch+1) %10 == 0:
        # if True:
            cprint("[TEST]")
            Procedure.Test(dataset, Recmodel, epoch, w, multicore = world.config['multicore'])
        torch.save(Recmodel.state_dict(), weight_file)
finally:
    if world.tensorboard:
        w.close()

end = time.time()
time_elasped = end - start

print(f'Time elasped: {str(timedelta(minutes=end - start))}')