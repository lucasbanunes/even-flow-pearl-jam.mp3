import json
from typing import Annotated, get_origin, get_args
from pydantic import BaseModel, Field
from pathlib import Path
from abc import ABC, abstractmethod

# --- Reuse your mock setup ---
try:
    import mlflow
except ImportError:
    import unittest.mock as mlflow

# --- Helper: Resolve Nested Models ---


def resolve_pydantic_model(annotation):
    """Finds the underlying Pydantic model inside List[], Annotated[], or Union[]."""
    if isinstance(annotation, type) and issubclass(annotation, BaseModel):
        return annotation

    origin = get_origin(annotation)
    args = get_args(annotation)

    if origin is Annotated:
        return resolve_pydantic_model(args[0])

    if origin in (list, tuple) or str(origin) == "typing.Union":
        for arg in args:
            model = resolve_pydantic_model(arg)
            if model:
                return model
    return None

# --- Helper: Clean Type String ---


def format_type_str(annotation):
    """Cleans up the type representation for the YAML output."""
    s = str(annotation).replace("typing.", "").replace(
        "<class '", "").replace("'>", "")
    if hasattr(annotation, "__name__"):
        return annotation.__name__
    # Handle Optional/Union visually
    return s.replace("Union[", "").replace(", NoneType]", "?").replace("]", "")

# --- The YAML Generator ---


def get_model_yaml(model_class: type[BaseModel], indent_level: int = 0) -> str:
    lines = []
    # YAML uses 2 spaces for indentation standard
    base_indent = "  " * indent_level
    prop_indent = "  " * (indent_level + 1)

    # If it's the root level, add the class docstring as a comment header
    if indent_level == 0 and model_class.__doc__:
        lines.append(f"# --- {model_class.__doc__.strip()} ---")
        lines.append("")

    for field_name, field_info in model_class.model_fields.items():
        # 1. Get Type and Description
        type_str = format_type_str(field_info.annotation)
        description = field_info.description or "No description"

        # 2. Print the Field Key
        lines.append(f"{base_indent}{field_name}:")

        # 3. Print Metadata
        lines.append(f'{prop_indent}type: "{type_str}"')
        # We wrap description in quotes to handle special YAML chars like ':'
        lines.append(f'{prop_indent}help: "{description}"')

        # 4. Handle Recursion (Nested Models)
        sub_model = resolve_pydantic_model(field_info.annotation)
        if sub_model:
            lines.append(f"{prop_indent}properties:")
            # Recursively call with +2 indentation level (key -> properties -> [children])
            lines.append(get_model_yaml(sub_model, indent_level + 2))

        lines.append("")  # Add spacing for readability

    return "\n".join(lines).strip()

# --- Example Setup ---


# 1. Your MLFlow Types
ID_TYPE_HELP = "Unique identifier for the job"
type IdType = Annotated[str | None, Field(description=ID_TYPE_HELP)]

NAME_TYPE_HELP = "Name of the job"
type NameType = Annotated[str, Field(description=NAME_TYPE_HELP)]

# 2. A submodel for demonstration


class DatabaseConfig(BaseModel):
    "Very cool database config"
    host: str = Field(description="DB Host URL")
    port: int = Field(5432, description="DB Port")

# 3. Your Main Model (Modified to include the submodel for demo)


class MLFlowBaseModel(BaseModel, ABC):
    """BaseModel with MLflow logging capabilities."""
    id_: IdType = None
    name: NameType = None
    db_config: DatabaseConfig = Field(
        default=None, description="Database settings")


# --- Run ---
yaml_docs = get_model_yaml(MLFlowBaseModel)
print(yaml_docs)
