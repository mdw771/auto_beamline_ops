import logging
from collections.abc import MutableMapping

import torch
import numpy as np
import torch.utils
import torch.utils.data
import generic_trainer

from autobl.steering.model import *
from torch.nn import MSELoss
import autobl.steering.io_util as io_util

try:
    import mlflow
    mlflow_ok = True
except ImportError:
    print('MLFlow is not installed. You can ignore this warning.')
    mlflow_ok = False

logging.basicConfig(format='[%(asctime)s] %(message)s', level=logging.INFO)


# =====================
use_mlflow = True & mlflow_ok
config_json = 'config_jsons/normalizedERD_convMLP_dimReconSpec1000_dimSpecEncoded256_dimFeatEncoded256_dimHiddenFeat128_dimHiddenFinal128_pooled_sigmoid.json'
resume_from_checkpoint = False
mlflow_name = "normalizedERD, convMLP, dimReconSpec=1000, dimSpecEncoded=256, dimFeatEncoded=256, dimHiddenFeat128, dimHiddenFinal128, feat=pos+3NNs, pooled, sigmoid, gaussRandEnc, lr=1e-3"
mlflow_id = None
model_save_dir = 'trained_models/testmodel_normalizedERD_convMLP_dimReconSpec1000_dimSpecEncoded256_dimFeatEncoded256_dimHiddenFeat128_dimHiddenFinal128_pooled_sigmoid_featPos3NNs_gaussRandEnc_lr1e-3'
# =====================


def plot_erds():
    if trainer.current_epoch == trainer.configs.num_epochs - 1:
        import matplotlib.pyplot as plt
        n = 5
        fig, ax = plt.subplots(5, 1, figsize=(3.5, 1.5 * n))
        for ip in range(n):
            data = []
            for i in range(157 * ip, 157 * (ip + 1)):
                data.append(trainer.dataset[i])
            x = torch.cat([a[0] for a in data], dim=0).cuda()
            x_meas = torch.cat([a[1] for a in data], dim=0).cuda()
            y_interp = torch.cat([a[2] for a in data], dim=0).cuda()
            erd = torch.cat([a[3] for a in data], dim=0).cuda()
            preds = trainer.model(x, x_meas, y_interp)
            ax[ip].plot(x.squeeze().cpu().detach().numpy(), erd.squeeze().cpu().detach().numpy(), label='True ERD')
            ax[ip].plot(x.squeeze().cpu().detach().numpy(), preds.squeeze().cpu().detach().numpy(), label='Predicted ERD')
            plt.legend()
        plt.tight_layout()
        mlflow.log_figure(fig, 'erds.png')

if use_mlflow:
    def flatten(dictionary, parent_key='', separator='_'):
        items = []
        for key, value in dictionary.items():
            new_key = parent_key + separator + key if parent_key else key
            if isinstance(value, MutableMapping):
                items.extend(flatten(value, new_key, separator=separator).items())
            else:
                items.append((new_key, value))
        return dict(items)

    def log_loss_to_mlflow_hook():
        mlflow.log_metrics({
            'loss': trainer.loss_tracker['loss'][-1],
            }, 
            step=trainer.loss_tracker['epochs'][-1]
        )
        
    def log_val_loss_to_mlflow_hook():
        mlflow.log_metrics({
            'val_loss': trainer.loss_tracker['val_loss'][-1],
            }, 
            step=trainer.loss_tracker['epochs'][-1]
        )
        
        plot_erds()


dataset = io_util.SLADSDataset("slads_data/data_train_normalized.h5", 
                               returns=("x", "x_measured", "y_interp", "erd"),
                               n_recon_pixels=1000)
# Uncomment to shrink dataset for debugging
# dataset = torch.utils.data.Subset(dataset, range(100))

configs = generic_trainer.configs.TrainingConfig()
configs.load_from_json(config_json, namespace=globals())
configs.dataset = dataset

if use_mlflow:
    configs.post_training_epoch_hook = log_loss_to_mlflow_hook
    configs.post_validation_epoch_hook = log_val_loss_to_mlflow_hook
    mlflow.set_tracking_uri(uri="http://164.54.100.71:5000")
    mlflow.set_experiment('slads_xanes')
    
    mlflow_kwargs = {
        'run_name': mlflow_name,
    }
    if mlflow_id is not None:
        mlflow_kwargs['run_id'] = mlflow_id
        
    mlflow.start_run(**mlflow_kwargs)
    if mlflow_id is None:
        # Attempting to log existing params will trigger an Exception (and it will be wrongly stated that the exception 
        # is an HTTP connection error which is not the case).
        mlflow.log_params(flatten(configs.get_serializable_dict()))
        mlflow.log_artifact(__file__)

trainer = generic_trainer.Trainer(configs)
trainer.build()
if use_mlflow:
    mlflow.log_metric('num_params', np.sum([torch.numel(w) if w.requires_grad else 0 for w in trainer.model.parameters()]))
trainer.run_training()

if use_mlflow:
    mlflow.end_run()