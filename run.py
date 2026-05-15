import hydra

from hydra.utils import instantiate

from omegaconf import DictConfig

from src.experiment import Experiment



@hydra.main(
    version_base=None,
    config_path="configs",
    config_name="config",
)
def main(
    config: DictConfig,
):

    logger = instantiate(
        config.logger,
    )

    logger.save_config(
        config,
    )

    function = instantiate(
        config.function,
    )

    optimizer = instantiate(
        config.optimizer,
        logger=logger,
    )

    experiment = Experiment(
        function=function,
        optimizer=optimizer,
        logger=logger,
        config=config,
    )

    experiment.run()


if __name__ == "__main__":
    main()
