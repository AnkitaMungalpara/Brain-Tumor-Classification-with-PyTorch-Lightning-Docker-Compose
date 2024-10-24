import argparse
import os

import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger

from datamodules.brain_tumor import BrainTumorDataModule
from models.brain_tumor_classifier import BrainTumorClassifier
from utils.pylogger import get_pylogger

log = get_pylogger(__name__)


def train(args):

    # 1. data module
    data_module = BrainTumorDataModule(data_dir="data", batch_size=32)

    # 2. set up model
    if os.path.exists(args.ckpt_path):
        model = BrainTumorClassifier.load_from_checkpoint(args.ckpt_path)
    else:
        model = BrainTumorClassifier(lr=1e-3)

    # 3. define logger
    logger = TensorBoardLogger(save_dir="logs", name="brain_tumor_classifier")

    # 4. set up callback
    checkpoint_callback = ModelCheckpoint(
        monitor="val/loss",
        dirpath="model/",
        filename="brain_tumor_classifier_model",
        save_top_k=1,
        mode="min",
    )

    # 5. Define Trainer
    trainer = L.Trainer(
        max_epochs=2,
        callbacks=[checkpoint_callback],
        logger=logger,
        log_every_n_steps=1,
        accelerator="auto",
    )

    config = {"data": vars(data_module), "model": vars(model), "trainer": vars(trainer)}
    log.info(config)

    # train the model
    if not os.path.exists(args.ckpt_path):
        log.info("Started model training as no checkpoint found.")
        trainer.fit(model=model, datamodule=data_module)

    # test the module
    trainer.test(model=model, datamodule=data_module)

    log.info("finishing up the training...")


if __name__ == "__main__":
    """
    Train the model using the provided arguments, including the model checkpoint path.
    """

    parser = argparse.ArgumentParser(description="Performs training")

    parser.add_argument(
        "--ckpt_path",
        type=str,
        default="checkpoints/brain_tumor_classifier_model.ckpt",
        help="path to the model checkpoint",
    )

    args = parser.parse_args()
    train(args)
