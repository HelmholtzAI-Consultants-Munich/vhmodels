# This file contains all the CLI functions available through the CLI (try vh-checker <command> <project_name>)
# With the CLI, one can list all available models
# and create conda envs and Apptainer images for the models

from vhmodels.registry import MODEL_REGISTRY

import click
import subprocess
import json
from pathlib import Path
from rich.console import Console
from rich.table import Table


@click.group()
def main():
    """vhmodels: Manage and run isolated genomic models."""
    pass


console = Console()


@main.command()
def list():
    """Display all registered models and their descriptions."""

    table = Table(
        title="Available models",
        header_style="bold magenta",
        border_style="bright_black",
    )

    table.add_column("ID", style="cyan", no_wrap=True)
    table.add_column("Model Name", style="bold green")
    table.add_column("Description", style="white")

    for i, (name, desc) in enumerate(MODEL_REGISTRY.items(), 1):
        table.add_row(str(i), name, desc["description"])

    console.print(table)


def _check_conda_installed():
    """Check if Conda is available"""
    try:
        subprocess.run(
            ["conda", "--version"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
        )
        return True
    except FileNotFoundError:
        click.echo("Error: Conda is not installed or not found in PATH.")
        return False
    except subprocess.CalledProcessError:
        click.echo(
            "Error: Conda is installed but returned an error. Check your Conda installation."
        )
        return False


@main.command()
@click.argument("project")
def create_env(project):
    """Create a Conda environment for a specific model."""
    # Check if Conda is installed
    if not _check_conda_installed():
        return

    if project not in MODEL_REGISTRY.keys():
        click.echo(f"Error: Model '{project}' is not registered.")
        return

    # model_cls = BaseModel.get_class(project)
    env_name = f"vhmodels-{project}"  # model_cls.env_name

    # Locate the model directory dynamically (handling potential capitalization)
    models_dir = Path(__file__).parent.parent / "models"
    # Find the folder that matches the ID (case-insensitive)
    target_dir = next(
        (
            d
            for d in models_dir.iterdir()
            if d.is_dir() and d.name.lower() == project.lower()
        ),
        None,
    )

    if not target_dir:
        click.echo(f"Error: Could not find directory for project {project}")
        return

    env_file = target_dir / "environment.yml"

    if not env_file.exists():
        click.echo(f"Error: environment.yml not found at {env_file}")
        return

    click.echo(f"Creating environment '{env_name}'...")
    try:
        subprocess.run(
            ["conda", "env", "create", "-n", env_name, "-f", str(env_file)], check=True
        )
        click.echo(f"Successfully created {env_name}.")
    except subprocess.CalledProcessError:
        click.echo(
            "Failed to create environment. Ensure Conda is installed and functional."
        )


# TODO: Implement function the creates the Apptainer image


@main.command()
@click.argument("model")
@click.argument("data_json")
def run(model_id, data_json):
    """Run a model directly from the CLI using a JSON string as input."""
    from vhmodels.vh_checker.factory import load_model

    try:
        data = json.loads(data_json)
        model = load_model(model_id)
        result = model.embed(data)
        click.echo(json.dumps(result, indent=2))
    except Exception as e:
        click.echo(f"Error: {str(e)}")


if __name__ == "__main__":
    main()
