import os
import time
import logging
from tqdm import tqdm
from datetime import datetime
from collections import defaultdict

import numpy as np
import torch

from .prepare import prepare, prepare_classifier

from ..evaluate.evaluate import evaluate, evaluate_loss
from ..utils.general_utils import MyLogger, ljson
from ..utils.data_utils import move_tensors

def train(args, prepare=prepare, evaluate=evaluate):
    """
    Trains a VCI model
    """
    # SETUP ---------------------
    if args["seed"] is not None:
        np.random.seed(args["seed"])
        torch.manual_seed(args["seed"])
    device = (
        "cuda:" + str(args["gpu"])
            if (not args["cpu"]) 
                and torch.cuda.is_available() 
            else 
        "cpu"
    )
    
    # LOAD MODEL & DATASET ---------------------
    state_dict = None
    if args["checkpoint"] is not None:
        state_dict, args = torch.load(args["checkpoint"], map_location="cpu")
    model, datasets = prepare(args, state_dict=state_dict, device=device)

    # LOGGING ---------------------
    dt = datetime.now().strftime("%Y.%m.%d_%H:%M:%S")
    log_dir = os.path.join(args["artifact_path"], "runs/" + args["name"] + "_" + dt)
    save_dir = os.path.join(args["artifact_path"], "saves/" + args["name"] + "_" + dt)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(save_dir, exist_ok=True)
    print(args)
    logger = MyLogger(args, log_dir, name=args["name"], is_wandb=args["use_wandb"])
    

    ljson({"training_args": args})
    ljson({"model_params": model.hparams})
    logging.info("")
    
    # TRAINING ---------------------
    start_time = time.time()
    for epoch in range(args["max_epochs"]):
        print(epoch)
        epoch_training_stats = defaultdict(float)
        for batch_idx, batch in tqdm(enumerate(datasets["train_loader"]), total=len(datasets["train_loader"])):
            minibatch_training_stats = model.update(move_tensors(*batch, device=device), batch_idx, logger)
            for key, val in minibatch_training_stats.items():
                epoch_training_stats[key] += val
        
        model.step()
        stop = (epoch == args["max_epochs"] - 1)    # stop if max epochs reached OR below early stopping threshold

        # Logging -- setup
        for key, val in epoch_training_stats.items():
            epoch_training_stats[key] = val / len(datasets["train_loader"])
        ellapsed_minutes = (time.time() - start_time) / 60
        all_epoch_stats = {**epoch_training_stats, "ellapsed_minutes": ellapsed_minutes, "epoch": epoch}

        # Logging
        if (epoch % args["checkpoint_freq"]) == 0 or stop:
            #  Evaluation every checkpoint_freq epochs. TODO: decouple logging from evaluation from saving
            evaluation_stats, early_stop = evaluate(model, datasets,
                epoch=epoch, save_dir=save_dir, **args
            )
            all_epoch_stats.update(evaluation_stats)
            logger.update(all_epoch_stats, step=epoch, commit=True)

            # Save model
            torch.save(
                (model.state_dict(), args),
                os.path.join(save_dir, "model_seed={}_epoch={}.pt".format(args["seed"], epoch),),
            )
            ljson({"model_saved": "model_seed={}_epoch={}.pt\n".format(args["seed"], epoch)})

            # Early stopping
            if stop:
                ljson({"stop": epoch})
                break
            if early_stop:
                ljson({"early_stop": epoch})
                break
        else:
            # Log training stats every epoch. TODO: decouple logging from evaluation from saving, add flag for log_freq
            logger.update(all_epoch_stats, step=epoch, commit=True)

    logger.close()
    return model

def train_classifier(args, prepare=prepare_classifier, evaluate=evaluate_loss):
    return train(args, prepare=prepare, evaluate=evaluate)
