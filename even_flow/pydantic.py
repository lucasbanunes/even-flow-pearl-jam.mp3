from abc import ABC, abstractmethod
from typing import Annotated
from pathlib import Path
from typing import Self
from pydantic import BaseModel, Field, ConfigDict
from typer import Option
import yaml
import mlflow


class YamlBaseModel(BaseModel):

    @classmethod
    def from_yaml(cls, path: Path | str) -> Self:
        """Load a Pydantic model from a YAML file."""
        if isinstance(path, str):
            path = Path(path)

        with path.open("r") as f:
            data = yaml.safe_load(f)

        return cls(**data)


ID_TYPE_HELP = "Unique identifier for the job"
type IdType = Annotated[
    str | None,
    Field(
        description=ID_TYPE_HELP
    ),
    Option('--id', help=ID_TYPE_HELP)
]


NAME_TYPE_HELP = "Name of the job"
type NameType = Annotated[
    str,
    Field(
        description=NAME_TYPE_HELP
    ),
]

type RunType = Annotated[
    mlflow.entities.Run | None,
    Field(
        description="MLflow Run associated with the class."
    )
]


class MLFlowLoggedModel(BaseModel, ABC):

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @abstractmethod
    def _to_mlflow(self, prefix: str = '') -> None:
        raise NotImplementedError(
            "to_mlflow method must be implemented by subclasses.")

    def to_mlflow(self, prefix: str = '') -> None:
        self._to_mlflow(prefix)

    @classmethod
    @abstractmethod
    def from_mlflow(cls, mlflow_run: mlflow.entities.Run,
                    prefix: str = '') -> Self:
        raise NotImplementedError(
            "from_mlflow method must be implemented by subclasses.")

    @classmethod
    def from_mlflow_run_id(cls, run_id: str, prefix: str = '') -> Self:
        mlflow_client = mlflow.MlflowClient()
        mlflow_run = mlflow_client.get_run(run_id)
        instance = cls.from_mlflow(mlflow_run, prefix=prefix)
        instance.id_ = run_id
        instance.name = mlflow_run.data.tags.get('mlflow.runName', None)
        # instance.mlflow_run = mlflow_run
        return instance


def format_type(annotation):
    """Helper to format the type annotation into a readable string."""
    # This converts type objects to strings like 'str | None'
    if hasattr(annotation, "__name__"):
        return annotation.__name__
    return str(annotation).replace("typing.", "")


def get_model_docs(model_class: type[BaseModel]) -> str:
    lines = []

    # 1. Get the Class Docstring
    if model_class.__doc__:
        lines.append(model_class.__doc__.strip())
        lines.append("")  # Add empty line

    # 2. Iterate over model fields to get name, type, and description
    for field_name, field_info in model_class.model_fields.items():

        # Format the type (e.g., str | None)
        type_str = format_type(field_info.annotation)

        # Get the description defined in Field()
        description = field_info.description or "No description provided."

        # Format: "name: type"
        lines.append(f"{field_name}: {type_str}")
        # Format: "    Description"
        lines.append(f"    {description}")
        lines.append("")  # Add empty line between fields

    return "\n".join(lines).strip()
