import os
from pathlib import Path



def get_project_root() -> Path:
    """
    Find the project root directory (works across machines).
    """
    current = Path(__file__).resolve()
    for parent in current.parents:
        return parent


def prepare_workspace():
    """
    Prepares the workspace by creating necessary directories under the project root/bin
    and initializing logs.
    """
    root = get_project_root()
    bin_dir = root / "bin"

    # Directories to create
    dirs = [
        bin_dir,
        bin_dir / "data_outputs",
        bin_dir / "models_deployed",
        bin_dir / "artifacts",
        bin_dir / "docs",
        bin_dir / "mlruns",
        bin_dir / "scripts",

    ]

    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)

    set_credentials()

    return root


def set_credentials():
    import dotenv
    dotenv.load_dotenv()


if __name__ == "__main__":
    prepare_workspace()
