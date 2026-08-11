"""Command-line interface for listing, preparing, and running models."""

import json
import os
import platform
import shutil
import subprocess
import tempfile
from pathlib import Path

import click
from rich.console import Console
from rich.table import Table

from vhmodels.registry import MODEL_REGISTRY
from vhmodels.utils import lima_utils


@click.group()
def main():
    """vhmodels: Manage and run isolated genomic models."""
    pass


console = Console()

_APPTAINER_UV_CACHE_DEST = "/opt/vhmodels-build-cache"


def _apptainer_uv_cache_dir():
    """Return the persistent, per-user cache used by uv image builds."""
    configured = os.environ.get("VHMODELS_APPTAINER_CACHE_DIR")
    if configured:
        return Path(configured).expanduser().resolve()

    cache_home = os.environ.get("XDG_CACHE_HOME")
    cache_root = Path(cache_home).expanduser() if cache_home else Path.home() / ".cache"
    return (cache_root / "vhmodels" / "apptainer" / "uv").resolve()


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


def _check_pip_installed(env_name):
    result = subprocess.run(
        ["conda", "run", "-n", env_name, "python", "-m", "pip", "--version"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return result.returncode == 0


def _check_apptainer_installed():
    """Check if Apptainer is available."""
    try:
        subprocess.run(
            ["apptainer", "--version"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
        )
        return True
    except FileNotFoundError:
        raise click.ClickException("Apptainer is not installed or not found in PATH.")
    except subprocess.CalledProcessError:
        raise click.ClickException(
            "Apptainer is installed but returned an error. "
            "Check your Apptainer installation."
        )


def _determine_current_platform():
    system_map = {"Linux": "linux", "Darwin": "macos"}
    current_system = system_map.get(platform.system(), platform.system().lower())
    current_machine = platform.machine()
    return f"{current_system}-{current_machine}"


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

    supported_platforms = MODEL_REGISTRY[project]["supported_platforms"]
    current_platform = _determine_current_platform()
    if current_platform not in supported_platforms:
        click.echo(
            f"Error: current platform '{current_platform}' is not supported; supported platforms are: {', '.join(supported_platforms)}"
        )
        return

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

    env_file = MODEL_REGISTRY[project]["environment_files"][current_platform]
    env_path = target_dir / env_file

    if not env_path.exists():
        click.echo(f"Error: environment.yml not found at {env_path}")
        return

    click.echo(f"Creating environment '{env_name}'...")
    try:
        subprocess.run(
            ["conda", "env", "create", "-n", env_name, "-f", str(env_path)], check=True
        )
        # install vhmodels for communication with host
        if not _check_pip_installed(env_name):
            click.echo(
                f"pip not installed via '{env_path}'. Additionally installing pip..."
            )
            subprocess.run(
                ["conda", "install", "-y", "-n", env_name, "pip"],
                check=True,
            )
        click.echo(f"Installing vhmodels into '{env_name}'...")
        project_root = Path(__file__).resolve().parent.parent.parent
        subprocess.run(
            [
                "conda",
                "run",
                "-n",
                env_name,
                "python",
                "-m",
                "pip",
                "install",
                "-e",
                str(project_root),
            ],
            check=True,
        )

        click.echo(f"Successfully created {env_name}.")
    except subprocess.CalledProcessError:
        click.echo(
            "Failed to create environment. Ensure Conda is installed and functional."
        )


@main.command()
@click.argument("project")
@click.option(
    "--output",
    type=click.Path(dir_okay=False, path_type=Path),
    help="Output SIF path (default: vhmodels-<project>.sif).",
)
def create_apptainer_image(project, output):
    """Create an Apptainer image for a specific model."""
    if project not in MODEL_REGISTRY:
        raise click.ClickException(f"Model '{project}' is not registered.")

    host_platform = _determine_current_platform()
    use_lima = host_platform.startswith("macos-")
    if not use_lima and not host_platform.startswith("linux-"):
        raise click.ClickException(
            "Apptainer images can only be built on Linux or on macOS with Lima."
        )
    # macOS uses a native ARM Lima VM plus Rosetta to build the same AMD64
    # Linux image that runs natively on the supported HPC platform.
    target_platform = "linux-x86_64" if use_lima else host_platform

    supported_platforms = MODEL_REGISTRY[project]["supported_platforms"]
    if target_platform not in supported_platforms:
        raise click.ClickException(
            f"Target platform '{target_platform}' is not supported; "
            f"supported platforms are: {', '.join(supported_platforms)}"
        )

    package_dir = Path(__file__).resolve().parent.parent
    template_path = package_dir / "envs" / "Apptainer"
    target_dir = Path(MODEL_REGISTRY[project]["abs_path"])
    apptainer_config = MODEL_REGISTRY[project].get("apptainer", {})
    python_version = apptainer_config.get("python")
    requirements_file = apptainer_config.get("requirements")
    exclude_file = apptainer_config.get("exclude")
    torch_backend = apptainer_config.get("torch_backend")
    requirements_path = target_dir / requirements_file if requirements_file else None
    exclude_path = target_dir / exclude_file if exclude_file else None

    if not template_path.is_file():
        raise click.ClickException(
            f"Apptainer definition template not found at {template_path}"
        )
    if not python_version or requirements_path is None:
        raise click.ClickException(
            f"Apptainer dependency configuration is missing for project '{project}'."
        )
    if not requirements_path.is_file():
        raise click.ClickException(
            f"Apptainer requirements file not found at {requirements_path}"
        )
    if exclude_path is not None and not exclude_path.is_file():
        raise click.ClickException(
            f"Apptainer exclude file not found at {exclude_path}"
        )

    image_path = Path(output or f"vhmodels-{project}.sif").expanduser().resolve()
    if image_path.exists():
        raise click.ClickException(
            f"Output image '{image_path}' already exists; choose another "
            "path or remove it first."
        )

    if not image_path.parent.is_dir():
        raise click.ClickException(
            f"Output directory '{image_path.parent}' does not exist."
        )
    if use_lima and not lima_utils.is_lima_shared_path(image_path):
        raise click.ClickException(
            "On macOS, the output image must be under your home directory "
            "so Lima can access it."
        )

    uv_cache_path = _apptainer_uv_cache_dir()
    if use_lima and not lima_utils.is_lima_shared_path(uv_cache_path):
        raise click.ClickException(
            "On macOS, the Apptainer build cache must be under your home "
            "directory so Lima can access it. Set "
            "VHMODELS_APPTAINER_CACHE_DIR to a path under your home directory."
        )

    if use_lima:
        try:
            lima_utils.ensure_lima_instance()
        except RuntimeError as error:
            raise click.ClickException(str(error)) from error
    else:
        _check_apptainer_installed()

    try:
        uv_cache_path.mkdir(parents=True, exist_ok=True)
    except OSError as error:
        raise click.ClickException(
            f"Could not create Apptainer build cache at '{uv_cache_path}'."
        ) from error

    click.echo(f"Using uv build cache '{uv_cache_path}'.")

    definition = template_path.read_text(encoding="utf-8")
    replacements = {
        "{project}": project,
        "{model_dir}": target_dir.name,
        "{python_version}": python_version,
        "{requirements_file}": requirements_file,
        "{torch_backend_arg}": (
            f"--torch-backend {torch_backend}" if torch_backend else ""
        ),
        "{exclude_arg}": (
            f"--excludes /opt/vhmodels-src/vhmodels/models/{target_dir.name}/{exclude_file}"
            if exclude_file
            else ""
        ),
    }
    for placeholder, value in replacements.items():
        definition = definition.replace(placeholder, value)

    click.echo(f"Building Apptainer image '{image_path}'...")
    try:
        temp_parent = image_path.parent if use_lima else None
        with tempfile.TemporaryDirectory(
            prefix="vhmodels-apptainer-", dir=temp_parent
        ) as temp_dir:
            build_context = Path(temp_dir)
            shutil.copytree(
                package_dir,
                build_context / "vhmodels",
                ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
            )
            definition_path = build_context / "Apptainer.def"
            definition_path.write_text(definition, encoding="utf-8")
            build_command = ["apptainer", "build"]
            if use_lima:
                # Lima mounts /tmp as a small RAM disk. Large model images can
                # fill it while Apptainer assembles the uncompressed rootfs, so
                # place build scratch data on the VM's persistent disk instead.
                build_command = [
                    "env",
                    "APPTAINER_TMPDIR=/var/tmp",
                    *build_command,
                    "--arch",
                    "amd64",
                ]
            build_command += [
                "--bind",
                f"{uv_cache_path}:{_APPTAINER_UV_CACHE_DEST}",
                str(image_path),
                str(definition_path),
            ]
            if use_lima:
                build_command = lima_utils.lima_shell_command(
                    build_command, build_context
                )
            subprocess.run(
                build_command,
                check=True,
                cwd=build_context,
            )
        click.echo(f"Successfully created Apptainer image '{image_path}'.")
    except subprocess.CalledProcessError as error:
        raise click.ClickException("Failed to build Apptainer image.") from error


@main.command()
@click.argument("model_id")
@click.argument("data_json")
@click.option(
    "--runtime",
    type=click.Choice(["conda", "apptainer"]),
    default="conda",
    show_default=True,
)
@click.option(
    "--image-path",
    type=click.Path(dir_okay=False, path_type=Path),
    help="Path to a custom Apptainer SIF image.",
)
def run(model_id, data_json, runtime, image_path):
    """Run a model directly from the CLI using a JSON string as input."""
    from vhmodels.vh_checker.factory import load_model

    try:
        data = json.loads(data_json)
        model = load_model(model_id, runtime=runtime, image_path=image_path)
        result = model.embed(data)
        click.echo(json.dumps(result, indent=2))
    except Exception as e:
        click.echo(f"Error: {str(e)}")


if __name__ == "__main__":
    main()
