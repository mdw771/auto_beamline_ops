import logging
from collections.abc import MutableMapping

import torch
import numpy as np
import torch.utils
import torch.utils.data
import generic_trainer

import autobl.steering.model as models
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
resume_from_checkpoint = False
mlflow_name = "convMLP, dimReconSpec=1000, dimSpecEncoded=256, dimFeatEncoded=256, feat=pos+3NNs, gaussRandEnc, lr=1e-3"
mlflow_id = None
model_save_dir = 'trained_models/model_convMLP_dimReconSpec1000_dimSpecEncoded256_dimFeatEncoded256_featPos3NNs_gaussRandEnc_lr1e-3'
# =====================


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


dataset = io_util.SLADSDataset("slads_data/data_train.h5", 
                               returns=("x", "x_measured", "y_interp", "erd"),
                               n_recon_pixels=1000)
# Uncomment to shrink dataset for debugging
# dataset = torch.utils.data.Subset(dataset, range(100))

model_params = generic_trainer.configs.ModelParameters()
model_params.dim_recon_spec = 1000
model_params.dim_spec_encoded=256
model_params.dim_feat_encoded=256
model_params.add_pooling=False

configs = generic_trainer.configs.TrainingConfig(
    model_class=models.ConvMLPModel,
    model_params=model_params,
    dataset=dataset,
    data_label_separation_index=3,
    validation_ratio=0.1,
    pred_names_and_types=[["erd", "regr"]],
    cpu_only=False,
    random_seed=42,
    batch_size_per_process=32,
    num_epochs=120,
    learning_rate_per_process=1e-3,
    optimizer=torch.optim.Adam,
    model_save_dir="trained_models/model_test",
    loss_function=torch.nn.MSELoss()
)

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
        #mlflow.log_artifact(__file__)

trainer = generic_trainer.Trainer(configs)
trainer.build()
if use_mlflow:
    mlflow.log_metric('num_params', np.sum([torch.numel(w) if w.requires_grad else 0 for w in trainer.model.parameters()]))
trainer.run_training()

if use_mlflow:
    mlflow.end_run()