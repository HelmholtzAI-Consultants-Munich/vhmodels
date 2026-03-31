# This file contains all the CLI functions available through the CLI (try vh-checker <command> <project_name>)
# With the CLI, one can list all available models 
# and create conda envs, Docker images and Apptainer images for the models

from vhmodels.vh_checker.base import BaseModel
from vhmodels.registry import MODEL_REGISTRY

import click
import subprocess
import os
import json
from pathlib import Path
import io
import tarfile
import platform
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from pathlib import Path
import tempfile
import shutil

@click.group()
def main():
    """vhmodels: Manage and run isolated genomic models."""
    pass

console = Console()

@main.command()
def list():
    """Display all registered models and their descriptions."""

    table = Table(title="Available models", header_style="bold magenta", border_style="bright_black")
    
    table.add_column("ID", style="cyan", no_wrap=True)
    table.add_column("Model Name", style="bold green")
    table.add_column("Description", style="white")

    for i, (name, desc) in enumerate(MODEL_REGISTRY.items(), 1):
        table.add_row(str(i), name, desc['description'])

    console.print(table)

def _check_conda_installed():
    """Check if Conda is available"""
    try:
        subprocess.run(
            ["conda", "--version"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True
        )
        return True
    except FileNotFoundError:
        click.echo("Error: Conda is not installed or not found in PATH.")
        return False
    except subprocess.CalledProcessError:
        click.echo("Error: Conda is installed but returned an error. Check your Conda installation.")
        return False

@main.command()
@click.argument('project')
def create_env(project):
    """Create a Conda environment for a specific model."""
     # Check if Conda is installed 
    if not _check_conda_installed():
        return

    if project not in MODEL_REGISTRY.keys():
        click.echo(f"Error: Model '{project}' is not registered.")
        return

    #model_cls = BaseModel.get_class(project)
    env_name = f'vhmodels-{project}' #model_cls.env_name
    
    # Locate the model directory dynamically (handling potential capitalization)
    models_dir = Path(__file__).parent.parent / "models"
    # Find the folder that matches the ID (case-insensitive)
    target_dir = next((d for d in models_dir.iterdir() if d.is_dir() and d.name.lower() == project.lower()), None)

    if not target_dir:
        click.echo(f"Error: Could not find directory for project {project}")
        return

    env_file = target_dir / "environment.yml"
    
    if not env_file.exists():
        click.echo(f"Error: environment.yml not found at {env_file}")
        return

    click.echo(f"Creating environment '{env_name}'...")
    try:
        subprocess.run(["conda", "env", "create", "-n", env_name, "-f", str(env_file)], check=True)
        click.echo(f"Successfully created {env_name}.")
    except subprocess.CalledProcessError:
        click.echo("Failed to create environment. Ensure Conda is installed and functional.")

def _check_docker_installed():
    """Check if Docker is installed and in PATH."""
    try:
        subprocess.run(
            ["docker", "--version"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True
        )
        return True
    except FileNotFoundError:
        click.echo("Error: Docker is not installed or not found in PATH.")
        return False
    except subprocess.CalledProcessError:
        click.echo("Error: Docker is installed but returned an error. Check your Docker installation.")
        return False

@main.command()
@click.argument('project')
def create_docker_image(project):
    """Build Docker image dynamically in-memory for a specific project."""

    # Docker check
    if not _check_docker_installed():
        return 

    image_name = f"vhmodels-{project}"
    project_root = Path(__file__).parent.parent.parent

    # Paths
    dockerfile_template_path = project_root / "vhmodels" / "envs" / "Dockerfile"
    env_yml_path = project_root / "vhmodels" / "models" / project / "environment.yml"

    if not dockerfile_template_path.exists() or not env_yml_path.exists():
        click.echo("Dockerfile template or environment.yml not found.")
        return

    # Read Dockerfile template
    dockerfile_str = dockerfile_template_path.read_text()
    dockerfile_str = dockerfile_str.replace("{project}", project)

    # Create in-memory tar context
    context = io.BytesIO()
    with tarfile.open(fileobj=context, mode="w") as tar:
        # Add Dockerfile
        df_bytes = dockerfile_str.encode("utf-8")
        df_info = tarfile.TarInfo(name="Dockerfile")
        df_info.size = len(df_bytes)
        tar.addfile(df_info, io.BytesIO(df_bytes))

        # Add environment.yml at root
        env_bytes = env_yml_path.read_bytes()
        env_info = tarfile.TarInfo(name="environment.yml")
        env_info.size = len(env_bytes)
        tar.addfile(env_info, io.BytesIO(env_bytes))

        # Add all other project files
        for path in project_root.rglob("*"):
            if path.is_file() and not path.name.startswith(".") and "vhmodels.egg-info" not in path.parts:
                arcname = path.relative_to(project_root)
                tar.add(str(path), arcname=str(arcname))

    context.seek(0)

    # Build the image
    try:
        subprocess.run(
            ["docker", "build", "-t", image_name, "-"],
            input=context.read(),
            check=True,
            text=False
        )
        click.echo(f"Successfully created Docker image '{image_name}'.")
    except subprocess.CalledProcessError as e:
        click.echo("Failed to build Docker image.")
        click.echo(e.stderr)
        
# TODO: Implement function the creates the Apptainer image

@main.command()
@click.argument('model')
@click.argument('data_json')
def run(model_id, data_json):
    """Run a model directly from the CLI using a JSON string as input."""
    from vhmodels.vh_checker.factory import load_model
    
    try:
        data = json.loads(data_json)
        model = load_model(model_id)
        result = model.transform(data)
        click.echo(json.dumps(result, indent=2))
    except Exception as e:
        click.echo(f"Error: {str(e)}")

if __name__ == "__main__":
    main()