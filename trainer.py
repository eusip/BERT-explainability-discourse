import argparse
import glob
import logging
import os
import re
from pathlib import Path
import warnings

import torch

from pytorch_lightning.loggers import TensorBoardLogger

from lightning_base import (
    BaseTransformer,
    add_generic_args,
    add_trainer_args,
    generic_train,
)

from models import (
    AdaptiveFineTuning,
    MaskedLM,
    NextSentencePrediction,
    MLM_NSP,
    ProbingTask,
)

from data import (
    MonologueDialogue,
    DialogueMonologue,
    # DNCMonologuePresDialogue,
    # DNCMonologuePresMonologue,
    DNCForNSP,
    PresForNSP,
    DNCForClassification,
    PresForClassification,
)

# logger = logging.getLogger(__name__)
logger = logging.getLogger('trainer')

DATASETS = {
    "monologue_dialogue": MonologueDialogue,
    "dialogue_monologue": DialogueMonologue,
    # "dnc_presidential": DNCMonologuePresDialogue,
    # "dnc_presidential_monologue": DNCMonologuePresMonologue,
    "dnc_for_NSP": DNCForNSP,
    "pres_for_NSP": PresForNSP,
    "dnc_for_classif": DNCForClassification,
    "pres_for_classif": PresForClassification,
}

MODELS = {
    "MLM": MaskedLM,
    "MLM_NSP": MLM_NSP,
    "NSP": NextSentencePrediction,
    "probe": ProbingTask,
    "AFT": AdaptiveFineTuning,
}

def main():
    # instantiate generic-trainer parser
    generic_trainer_parser = argparse.ArgumentParser(add_help=False)
    add_generic_args(generic_trainer_parser, os.getcwd())
    add_trainer_args(generic_trainer_parser, os.getcwd())

    # parse generic and trainer args
    generic_trainer_args = generic_trainer_parser.parse_known_args()[0]

    # general parser
    parser = argparse.ArgumentParser(parents=[generic_trainer_parser])

    # determine model specific args for experiment
    MODELS[generic_trainer_args.model].add_model_specific_args(parser, os.getcwd())

    # parse args
    args = parser.parse_args()

    # configure CUDA settings
    if args.gpus:
        os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
        torch.cuda.empty_cache()
        torch.cuda.memory_summary(device=None, abbreviated=False)

    # instantiate model for experiment
    model = MODELS[args.model](args)

    # instantiate dataset for experiment
    data = DATASETS[model.hparams.dataset_name](model.hparams)

    # instantiate Tensorboard
    tb_logs = os.path.join(os.getcwd(), model.hparams.log_dir)
    rdir = Path(tb_logs)
    rdir.mkdir(parents=True, exist_ok=True)
    log_name = model.hparams.model + "_" + model.hparams.dataset_name
    tb_logger = TensorBoardLogger(save_dir=rdir, name=log_name)

    # instantiate trainer
    trainer = generic_train(model, args, logger=tb_logger)
    if args.do_train:
        trainer.fit(model, data)
    if args.auto_lr_find:
        trainer.tune(model, data)
    if args.do_train and args.auto_lr_find:
        trainer.tune(model, data)
        trainer.fit(model, data)
    if args.do_validate:
        trainer.validate(model, data)
    if args.do_predict:
        trainer.predict(model, data)


if __name__ == "__main__":
    main()