import click
import subprocess
import os
import json
from pathlib import Path
from vhmodels.vh_checker.base import BaseModel
from vhmodels.vh_checker.base import REGISTRY

@click.group()
def main():
    """vhmodels: Manage and run isolated genomic models."""
    pass

@main.command()
def list():
    """Display all registered models and their descriptions."""
    # !!! Implement later


    #models = BaseModel.list_available_models()
    # if not models:
    #     click.echo("No models found. Ensure models are correctly registered in vhmodels/models/")
    #     return

    # click.echo(f"{'Model ID':<15} | {'Description'}")
    # click.echo("-" * 60)

    # for m in models:
    #      click.echo(f"{m['id']:<15} | {m['desc']}")

    # for m in models:
    #     # Use click.style to make the ID stand out
    #     model_id = click.style(m['id'], fg="cyan", bold=True)
    #     click.echo(f"ID: {model_id}")
        
    #     # Nicely indent the metadata
    #     click.echo(f"  Description : {m['desc']}")
    #     click.echo(f"  HF Link     : {m.get('link', 'N/A')}")
        
    #     # We can use pprint for just the complex bits if they exist, 
    #     # but for now, simple strings are cleaner:
    #     click.echo("-" * 60)

@main.command()
@click.argument('project')
def create_env(project):
    """Create a Conda environment for a specific model."""
    if project not in REGISTRY.keys():
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

    env_file = target_dir / "environment.yaml"
    
    if not env_file.exists():
        click.echo(f"Error: environment.yaml not found at {env_file}")
        return

    click.echo(f"Creating environment '{env_name}'...")
    try:
        subprocess.run(["conda", "env", "create", "-n", env_name, "-f", str(env_file)], check=True)
        click.echo(f"Successfully created {env_name}.")
    except subprocess.CalledProcessError:
        click.echo("Failed to create environment. Ensure Conda is installed and functional.")

@main.command()
@click.argument('model_id')
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