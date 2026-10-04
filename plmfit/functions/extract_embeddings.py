import torch
from plmfit.shared_utils import utils
from lightning import Trainer
from lightning.pytorch.loggers import TensorBoardLogger
from plmfit.models.lightning_model import LightningModel, PredictionWriter
from plmfit.models.fine_tuners import LowRankAdaptationFineTuner
from plmfit.shared_utils.checkpoint_utils import load_finetuned_backbone
from lightning.pytorch.strategies import DeepSpeedStrategy
from plmfit.shared_utils.deepspeed_utils import (
    estimate_zero3_model_states_mem_needs_all_live,
    use_deepspeed,
)
from lightning.pytorch.tuner import Tuner
from typing import Optional
import pandas as pd

def extract_embeddings(args, logger, data: Optional[pd.DataFrame] = None):

    # Load dataset
    data = utils.load_dataset(args.data_type) if data is None else data

    model = utils.init_plm(args.plm, logger, task="extract_embeddings")

    model.experimenting = False

    if args.model_path is not None:
        model = load_finetuned_plm(model, args, logger)

    model.set_layer_to_use(args.layer)
    model.py_model.reduction = args.reduction

    encs = model.categorical_encode(data)
    encs = torch.tensor(encs).clone().detach()

    logger.save_data(vars(args), "arguments")

    data_loader = utils.create_predict_data_loader(encs, batch_size=args.batch_size)

    model = LightningModel(
        model.py_model,
        plmfit_logger=logger,
        log_interval=100,
        experimenting=model.experimenting,
        train=False,
    )
    model.eval()
    lightning_logger = TensorBoardLogger(
        save_dir=logger.base_dir, version=0, name="lightning_logs"
    )

    if use_deepspeed():
        strategy = DeepSpeedStrategy(
            stage=3,
            offload_optimizer=True,
            offload_parameters=True,
            load_full_weights=True,
        )
        devices = args.gpus
    else:
        # On CPU, or on GPU without DeepSpeed, run on a single device
        strategy = "auto"
        devices = 1

    pred_writer = PredictionWriter(logger=logger, write_interval="epoch", split_size=args.split_size)

    trainer = Trainer(
        default_root_dir=logger.base_dir,
        logger=lightning_logger,
        enable_progress_bar=False,
        devices=devices,
        strategy=strategy,
        precision="16-mixed",
        callbacks=[pred_writer],
    )
    # tuner = Tuner(trainer)

    # # Auto-scale batch size by growing it exponentially (default)
    # tuner.scale_batch_size(model, mode="power")

    if use_deepspeed():
        estimate_zero3_model_states_mem_needs_all_live(
            model, num_gpus_per_node=int(args.gpus), num_nodes=1
        )

    output = trainer.predict(model=model, dataloaders=data_loader)

    return output


def load_finetuned_plm(model, args, logger):
    """
    Replace the pretrained weights of `model` with those of a PLM fine-tuned with PLMFit,
    so that the embeddings are extracted from the fine-tuned model.

    `args.model_path` is the checkpoint of the fine-tuning run; `args.ft_method`,
    `args.target_layers` and `args.lora_config_path` must be the ones of that run.
    """
    if args.ft_method == "lora":
        # Add LoRA to the model as in the fine-tuning run, load its weights and merge them
        # into the weights of the backbone
        fine_tuner = LowRankAdaptationFineTuner(
            logger=logger, lora_config_path=args.lora_config_path
        )
        model = fine_tuner.prepare_model(model, target_layers=args.target_layers)
        load_finetuned_backbone(model.py_model, args.model_path, logger)
        model.py_model = model.py_model.merge_and_unload()
    elif args.ft_method == "full":
        load_finetuned_backbone(model.py_model, args.model_path, logger)
    else:
        raise ValueError(
            "Embeddings can be extracted from a checkpoint (--model_path) of a model "
            "fine-tuned with LoRA or with full fine-tuning: pass the method used as "
            f"--ft_method ('lora' or 'full'), got '{args.ft_method}'."
        )
    return model
