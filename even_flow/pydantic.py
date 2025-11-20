from pathlib import Path
from typing import Self
from pydantic import BaseModel
import yaml


class YamlBaseModel(BaseModel):

    @classmethod
    def from_yaml(cls, path: Path | str) -> Self:
        """Load a Pydantic model from a YAML file."""
        if isinstance(path, str):
            path = Path(path)

        with path.open("r") as f:
            data = yaml.safe_load(f)

        return cls(**data)


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
