from pathlib import Path
from typing import Self
import shutil
from datetime import datetime, timezone
import mlflow.entities
from pydantic import ConfigDict
import typer
import lightning as L
from lightning.pytorch.loggers import MLFlowLogger
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
import mlflow

from even_flow.utils import get_logger

from . import (
    NeuralODE,
    AdjointType,
    SolverType,
    AToleranceType,
    RToleranceType,
    LearningRateType,
)
from ..pydantic import YamlBaseModel
from ..jobs import MLFlowBaseModel, ConfigOption,
from ..models.mlp import build_mlp, DimsType, ActivationsType
from ..datasets.spiral import (
    StableSpirals,
    InitialStateLowType,
    InitialStateHighType,
    NTimestampsType,
    TStepType,
    DecayType,
    FrequencyType,
    NoiseType,
    SamplesType,
    BatchSizeType,
    RandomState
)


DEFAULT_TRAINING_JOB_METRICS = {
    'train': {},
    'val': {},
    'test': {},
    'predict': {}
}


app = typer.Typer(
    help="Commands for Neural ODE generative models."
)


class MLPSpiralFit(MLFlowBaseModel, YamlBaseModel):

    model_config = ConfigDict(arbitrary_types_allowed=True)

    activations: ActivationsType
    datamodule: StableSpirals
    dims: DimsType
    
    # initial_state_low: InitialStateLowType
    # initial_state_high: InitialStateHighType
    # n_timestamps: NTimestampsType
    # test_samples: SamplesType
    # train_samples: SamplesType
    # t_step: TStepType
    # x_decay: DecayType
    # y_decay: DecayTypepytest.fixture
    # x_freq: FrequencyType
    # y_freq: FrequencyType
    # x_noise: NoiseType
    # y_noise: NoiseType

    adjoint: AdjointType = True
    atol: AToleranceType = 1e-6
    batch_size: BatchSizeType = 32
    rtol: RToleranceType = 1e-6
    learning_rate: LearningRateType = 1e-3
    # random_state: RandomState = None
    solver: SolverType = 'dopri5'

    model: NeuralODE | None = None
    metrics: dict[str, dict[str, float]] = DEFAULT_TRAINING_JOB_METRICS.copy()
    # spirals: StableSpirals | None = None

    def model_post_init(self, context):
        super().model_post_init(context)

        vector_field = build_mlp(
            input_dim=2,
            output_dim=2,
            hidden_dims=[64, 64],
            activation='tanh',
        )
        if self.model is None:
            self.model = NeuralODE(
                vector_field=vector_field,
                adjoint=self.adjoint,
                solver=self.solver,
                atol=self.atol,
                rtol=self.rtol,
                learning_rate=self.learning_rate,
            )
        # self.spirals = StableSpirals(
        #     initial_state_low=self.initial_state_low,
        #     initial_state_high=self.initial_state_high,
        #     n_timestamps=self.n_timestamps,
        #     t_step=self.t_step,
        #     x_decay=self.x_decay,
        #     y_decay=self.y_decay,
        #     x_freq=self.x_freq,
        #     y_freq=self.y_freq,
        #     x_noise=self.x_noise,
        #     y_noise=self.y_noise,
        #     train_samples=self.train_samples,
        #     test_samples=self.test_samples,
        #     batch_size=self.batch_size,
        #     random_state=self.random_state,
        # )

    def from_mlflow(self, mlflow_run) -> Self:
        raise NotImplementedError(
            "Loading TrainingJob from MLflow is not implemented yet.")

    def _run(self, tmp_dir: Path, run: mlflow.entities.Run):
        logger = get_logger()
        mlflow_client = mlflow.MlflowClient()
        experiment = mlflow_client.get_experiment(run.info.experiment_id)
        lightning_mlflow_logger = MLFlowLogger(
            run_name=self.name,
            run_id=self.id_,
            tracking_uri=mlflow.mlflow.get_tracking_uri(),
            experiment_name=experiment.name
        )

        checkpoint = ModelCheckpoint(
            monitor=self.monitor,  # Monitor a validation metric
            dirpath=self.checkpoints_dir,  # Directory to save checkpoints
            filename='best-model-{epoch:02d}-{val_max_sp:.2f}',
            save_top_k=3,
            mode="max",  # Save based on maximum validation accuracy
            save_on_train_epoch_end=False
        )

        callbacks = [
            EarlyStopping(
                monitor=self.monitor,
                patience=self.patience,
                mode="max",
                min_delta=1e-3
            ),
            checkpoint,
        ]
        trainer = L.Trainer(
            max_epochs=self.max_epochs,
            accelerator=self.accelerator,
            devices=1,
            logger=lightning_mlflow_logger,
            callbacks=callbacks,
            enable_progress_bar=False,
        )
        logger.info('Starting training process...')
        fit_start = datetime.now(timezone.utc)
        mlflow.log_metric('fit_start', fit_start.timestamp())
        trainer.fit(self.model, datamodule=self.datamodule)
        fit_end = datetime.now(timezone.utc)
        mlflow.log_metric('fit_end', fit_end.timestamp())
        mlflow.log_metric(
            "fit_duration", (fit_end - fit_start).total_seconds())
        self.model = NeuralODE.load_from_checkpoint(
            checkpoint.best_model_path)
        self.log_model(tmp_dir, checkpoint)
        logger.info('Training completed and model logged to MLFlow.')
        shutil.rmtree(str(self.checkpoints_dir))

    def log_model(self):
        raise NotImplementedError("Logging the model is not implemented yet.")


@app.command()
def mlp_spiral_fit(config_path: ConfigOption) -> MLPSpiralFit:
    job = MLPSpiralFit.from_yaml(config_path)
    job.run()
    return job
