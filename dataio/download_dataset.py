from pathlib import Path
import subprocess, sys, os, tarfile


def _normalize_source_path(source_path: str) -> str:
    """Helper to clean the source path for os.path.join."""
    # Remove leading/trailing slashes
    cleaned_path = source_path.strip(r"\/")
    # Replace any forward/backward slashes with the OS separator
    # and normalize (e.g., remove ".." or ".")
    return os.path.normpath(cleaned_path)


def download_openxlab_dataset(source_path, target_path) -> bool:
    """
    Downloads and extracts a dataset from OpenXLab,
    accounting for its nested directory structure.
    """

    # --- Define the actual data paths ---

    # The openxlab tool creates a slug from the repo name
    repo_name_slug = "omniobject3d___OmniObject3D-New"

    # Clean the source_path (e.g., "/raw/blender_renders_24_views")
    relative_source_path = _normalize_source_path(source_path)

    # This is the *actual* root directory where files are downloaded
    # e.g., .../data/omniobject3d___OmniObject3D-New/raw/blender_renders_24_views
    download_root = os.path.join(target_path, repo_name_slug, relative_source_path)

    # These are the final directories containing the tar files
    img_dir = os.path.join(download_root, "img")
    camera_dir = os.path.join(download_root, "camera")

    # --- Robust Check ---
    # Check if the final directories exist AND the tar files are gone
    img_exists = os.path.exists(img_dir)
    camera_exists = os.path.exists(camera_dir)

    img_tars_remain = list(Path(img_dir).glob("*.tar.gz"))
    camera_tars_remain = list(Path(camera_dir).glob("*.tar.gz"))

    if img_exists and camera_exists and not img_tars_remain and not camera_tars_remain:
        print(f"Extracted data found in {download_root}. Skipping download.")
        return True

    # --- Download Command ---
    command = [
        "openxlab",
        "dataset",
        "download",
        "--dataset-repo", "omniobject3d/OmniObject3D-New",
        "--source-path", source_path,
        "--target-path", target_path,
    ]

    print(f"Attempting to download dataset: {source_path.split('/')[-1]}")
    print(f"Running command: {' '.join(command)}\n")

    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=True,
            encoding='utf-8'
        )
        print("--- Download Complete (or files verified) ---")
        print(result.stdout)

        # --- Corrected Extraction Logic ---
        print("Extracting tar files (if any)...")

        # Extract files *into* the 'img' directory itself
        for tar_file in Path(img_dir).glob("*.tar.gz"):
            print(f"Extracting {tar_file.name} to {img_dir}...")
            with tarfile.open(tar_file, "r") as tar:
                tar.extractall(img_dir)  # Extract into the same folder
            os.remove(tar_file)

        # Extract files *into* the 'camera' directory itself
        for tar_file in Path(camera_dir).glob("*.tar.gz"):
            print(f"Extracting {tar_file.name} to {camera_dir}...")
            with tarfile.open(tar_file, "r") as tar:
                tar.extractall(camera_dir)  # Extract into the same folder
            os.remove(tar_file)
        print("Extraction complete.")

    except FileNotFoundError:
        print(f"Error: 'openxlab' command not found.", file=sys.stderr)
        print("Please make sure you have installed it with 'pip install openxlab'", file=sys.stderr)
        return False

    except subprocess.CalledProcessError as e:
        print(f"Error: The download command failed with code {e.returncode}", file=sys.stderr)
        print("\n--- STDOUT ---", file=sys.stderr)
        print(e.stdout, file=sys.stderr)
        print("\n--- STDERR ---", file=sys.stderr)
        print("\nEnsure you are logged in ('openxlab login') and the repo ID is correct.", file=sys.stderr)
        return False

    except Exception as e:
        print(f"An unexpected error occurred: {e}", file=sys.stderr)
        return False

    return True


class OmniObject3D:
    def __init__(self,
                 root_dir: str,
                 split: str = 'train',
                 view_24: bool = True,
                 max_objects: int | None = 50,
                 modalities = None,
                 view_window: int = 4,
                 view_sampling: str = 'uniform',
                 ):
        if modalities is None:
            modalities = {"rgb": True, "depth": False, "mask": False}
        self.root_dir = root_dir
        self.split = split
        self.view_24 = view_24
        self.max_objects = max_objects
        self.modalities = modalities
        self.view_window = view_window
        self.view_sampling = view_sampling

        if self.view_24:
            source_path = "/raw/blender_renders_24_views"
        else:
            source_path = "/raw/blender_renders"

        download_target_path = os.path.join(self.root_dir, "data")

        download_openxlab_dataset(source_path, download_target_path)

        repo_name_slug = "omniobject3d___OmniObject3D-New"
        relative_source_path = _normalize_source_path(source_path)

        self.data_path = os.path.join(
            download_target_path,
            repo_name_slug,
            relative_source_path
        )

dataset = OmniObject3D(root_dir="../", split="train", view_24=True, max_objects=1, modalities={"rgb": True, "depth": False, "mask": False})