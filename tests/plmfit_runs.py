"""
Support for the tests that run PLMFit end to end, through its command-line entry point.

The runs fine-tune the smallest supported PLM (ESM-2 8M) to classify the residues of a
handful of random sequences, so each of them takes a few seconds on a CPU. The tests are skipped when the weights of the
model are not available.
"""

import glob
import json
import os
import random
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd

import plmfit.__main__ as plmfit_main
from plmfit.logger import Logger
from plmfit.shared_utils import utils

PLM = "esm2_t6_8M_UR50D"
REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"

# Two per-residue classification tasks on the same sequences
THREE_CLASSES = "three_classes"
TWO_CLASSES = "two_classes"
SPLIT = "split"

TRAINING_PARAMETERS = {
    "learning_rate": 0.01,
    "epochs": 3,
    "batch_size": 4,
    "loss_f": "cross_entropy",
    "optimizer": "adam",
    "val_split": 0.2,
    "weight_decay": 0,
    "warmup_steps": 0,
    "gradient_clipping": False,
    "scheduler": False,
    "scaler": False,
    "gradient_accumulation": 1,
    "early_stopping": 10,
    "epoch_sizing": 1.0,
    "model_output": "logits",
}


class PlmfitRunTestCase(unittest.TestCase):
    """Sets up small datasets and head configurations, and runs PLMFit on them."""

    @classmethod
    def setUpClass(cls):
        try:
            utils.init_plm(PLM, None, task="extract_embeddings")
        except Exception as error:
            raise unittest.SkipTest(f"{PLM} is not available: {error}")

        cls.workdir = tempfile.mkdtemp()
        cls.default_dirs = (utils.data_dir, utils.config_dir)
        utils.set_data_dir(f"{cls.workdir}/data")
        utils.set_config_dir(f"{cls.workdir}/config")
        shutil.copytree(f"{REPO_DIR}/config/peft", f"{cls.workdir}/config/peft")
        os.makedirs(f"{cls.workdir}/config/training")

        rng = random.Random(0)
        sequences = [
            "".join(rng.choice(AMINO_ACIDS) for _ in range(rng.randint(8, 20)))
            for _ in range(24)
        ]
        data = pd.DataFrame(
            {
                "aa_seq": sequences,
                "len": [len(seq) for seq in sequences],
                SPLIT: ["train"] * 16 + ["validation"] * 4 + ["test"] * 4,
            }
        )
        for name, n_classes in ((THREE_CLASSES, 3), (TWO_CLASSES, 2)):
            labels = [
                str([AMINO_ACIDS.index(aa) % n_classes for aa in seq]) for seq in sequences
            ]
            os.makedirs(f"{cls.workdir}/data/{name}")
            data.assign(label=labels).to_csv(
                f"{cls.workdir}/data/{name}/{name}_data_full.csv", index=False
            )

    @classmethod
    def tearDownClass(cls):
        utils.set_data_dir(cls.default_dirs[0])
        utils.set_config_dir(cls.default_dirs[1])
        shutil.rmtree(cls.workdir)

    @classmethod
    def write_head_config(cls, name, n_classes, **training_parameters):
        """Write the configuration of a linear head for token classification and return its file name."""
        architecture = {
            "network_type": "linear",
            "output_dim": n_classes,
            "task": "token_classification",
            "dropout": 0,
        }
        training = {**TRAINING_PARAMETERS, "no_classes": n_classes, **training_parameters}
        config = {"architecture_parameters": architecture, "training_parameters": training}
        with open(f"{cls.workdir}/config/training/{name}.json", "w") as f:
            json.dump(config, f)
        return f"{name}.json"

    @classmethod
    def write_lora_config(cls, name, **changes):
        """Write a variant of the default LoRA configuration and return its path (relative to the config folder)."""
        with open(f"{cls.workdir}/config/peft/lora_config.json") as f:
            lora_config = {**json.load(f), **changes}
        with open(f"{cls.workdir}/config/peft/{name}.json", "w") as f:
            json.dump(lora_config, f)
        return f"peft/{name}.json"

    @classmethod
    def run_plmfit(cls, experiment_name, expect_failure=False, **arguments):
        """
        Run `python -m plmfit` with the given arguments and return the experiment directory.

        PLMFit logs the exceptions raised by a run instead of propagating them, so the
        outcome is read from the log: an AssertionError is raised if the run failed
        unexpectedly (or succeeded when a failure was expected).
        """
        experiment_dir = f"{cls.workdir}/output/{experiment_name}"
        argv = [
            "plmfit",
            f"--plm={PLM}",
            f"--experiment_name={experiment_name}",
            f"--experiment_dir={experiment_dir}",
            f"--output_dir={cls.workdir}/output",
        ]
        argv += [f"--{name}={value}" for name, value in arguments.items()]
        Logger._instance = None  # the logger is a singleton: start a new one for this run
        with patch.object(sys, "argv", argv):
            plmfit_main.main()

        log = cls.read_log(experiment_dir)
        failed = "Traceback (most recent call last)" in log
        if failed != expect_failure:
            raise AssertionError(f"Unexpected outcome of the run '{experiment_name}':\n{log}")
        return experiment_dir

    @staticmethod
    def read_log(experiment_dir):
        (log_file,) = glob.glob(f"{experiment_dir}/*.log")
        with open(log_file) as f:
            return f.read()
