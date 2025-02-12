import hydra
from omegaconf import DictConfig
from omegaconf import OmegaConf as oc
from easyeditor import EditTrainer, MENDTrainingHparams, ZsreDataset, SERACTrainingHparams


@hydra.main(version_base=None, config_path="hydra/experiments")
def main(args: DictConfig) -> None:
    if "MEND" in args.hparams_dir:
        training_hparams = MENDTrainingHparams.from_hparams(args.hparams_dir)
    elif "SERAC" in args.hparams_dir:
        training_hparams = SERACTrainingHparams.from_hparams(args.hparams_dir)
    else:
        raise ValueError("Invalid hparams_dir")

    train_ds = ZsreDataset(args.data.train, config=training_hparams)
    eval_ds = ZsreDataset(args.data.val, config=training_hparams)
    trainer = EditTrainer(
        config=training_hparams,
        train_set=train_ds,
        val_set=eval_ds
    )
    trainer.run()


if __name__ == "__main__":
    main()