import os
import time
from datetime import datetime
from collections import defaultdict

import numpy as np

import torch

from ..evaluate.evaluate import evaluate, evaluate_classic

from ..model.model import load_VCI

from ..dataset.dataset import load_dataset_splits

from ..utils.general_utils import pjson, sjson
from ..utils.data_utils import data_collate

def prepare(args, state_dict=None):
    """
    Instantiates model and dataset to run an experiment.
    """

    datasets = load_dataset_splits(
        args["data_path"],
        sample_cf=(True if args["dist_mode"] == "match" else False),
    )

    args["num_outcomes"] = datasets["training"].num_genes
    args["num_treatments"] = datasets["training"].num_perturbations
    args["num_covariates"] = datasets["training"].num_covariates

    model = load_VCI(args, state_dict)

    return model, datasets

def train(args):
    """
    Trains a VCI model
    """
    if args["seed"] is not None:
        np.random.seed(args["seed"])
        torch.manual_seed(args["seed"])

    model, datasets = prepare(args)

    datasets.update(
        {
            "loader_tr": torch.utils.data.DataLoader(
                datasets["training"],
                batch_size=args["batch_size"],
                shuffle=True,
                collate_fn=(lambda batch: data_collate(batch, nb_dims=1))
            )
        }
    )

    pjson({"training_args": args})
    pjson({"model_params": model.hparams})
    args["hparams"] = model.hparams

    dt = datetime.now().strftime("%Y.%m.%d_%H:%M:%S")
    save_dir = os.path.join(args["artifact_path"], "saves/" + args["name"] + "_" + dt)
    os.makedirs(save_dir, exist_ok=True)

    start_time = time.time()
    for epoch in range(args["max_epochs"]):
        epoch_training_stats = defaultdict(float)

        for data in datasets["loader_tr"]:
            (genes, perts, cf_genes, cf_perts, covariates) = (
            data[0], data[1], data[2], data[3], data[4:])

            minibatch_training_stats = model.update(
                genes, perts, cf_genes, cf_perts, covariates
            )

            for key, val in minibatch_training_stats.items():
                epoch_training_stats[key] += val

        for key, val in epoch_training_stats.items():
            epoch_training_stats[key] = val / len(datasets["loader_tr"])
            if not (key in model.history.keys()):
                model.history[key] = []
            model.history[key].append(epoch_training_stats[key])
        model.history["epoch"].append(epoch)

        ellapsed_minutes = (time.time() - start_time) / 60
        model.history["elapsed_time_min"] = ellapsed_minutes

        # decay learning rate if necessary
        # also check stopping condition: 
        # patience ran out OR max epochs reached
        stop = (epoch == args["max_epochs"] - 1)

        if (epoch % args["checkpoint_freq"]) == 0 or stop:
            if args["eval_mode"] == "native":
                evaluation_stats = evaluate(
                    model, datasets,
                    test_all=args["test_all"]
                )
            elif args["eval_mode"] == "classic":
                evaluation_stats = evaluate_classic(model, datasets)
            else:
                raise ValueError("eval_mode not recognized")

            for key, val in evaluation_stats.items():
                if not (key in model.history.keys()):
                    model.history[key] = []
                model.history[key].append(val)
            model.history["stats_epoch"].append(epoch)

            pjson(
                {
                    "epoch": epoch,
                    "training_stats": epoch_training_stats,
                    "evaluation_stats": evaluation_stats,
                    "ellapsed_minutes": ellapsed_minutes,
                }
            )

            torch.save(
                (model.state_dict(), args, model.history),
                os.path.join(
                    save_dir,
                    "model_seed={}_epoch={}.pt".format(args["seed"], epoch),
                ),
            )

            pjson(
                {
                    "model_saved": "model_seed={}_epoch={}.pt\n".format(
                        args["seed"], epoch
                    )
                }
            )
            stop = stop or model.early_stopping(np.mean(evaluation_stats["test"]))
            if stop:
                pjson({"early_stop": epoch})
                break

    return model
