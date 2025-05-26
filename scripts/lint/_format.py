import subprocess
from pathlib import Path
from tqdm import tqdm
import os

# Configuration
DIRECTORIES2LINELENGTH = {
    "../../python/hidet": 120,
    "../../tests": 120,
    "../../gallery": 100
}
BLACK_ARGS = [
    "python", "-m", "black",
    "--skip-string-normalization",
    "--skip-magic-trailing-comma",
    "--workers", str(os.cpu_count())
]


def main():
    # Determine the number of workers for black
    num_workers = os.cpu_count() or 1

    # Prepare commands for each directory
    commands = []
    for dir_path, line_length in DIRECTORIES2LINELENGTH.items():
        path = Path(dir_path)
        if path.exists() and path.is_dir():
            commands.append([
                *BLACK_ARGS,
                "--line-length", str(line_length),
                str(path)  # The directory to format
            ])
        else:
            tqdm.write(f"⚠️ Warning: Configured directory does not exist or is not a directory: {dir_path}")

    if not commands:
        tqdm.write("No valid directories found to format. Exiting.")
        return

    # Process each directory sequentially, but black itself will parallelize within each directory
    for command_args in commands:
        directory_to_format = os.path.abspath(command_args[-1])  # The last argument is the directory path
        # print the directory being formatted in green and bold font
        # CYAN = "\033[96m"
        BOLD = "\033[1m"
        RESET = "\033[0m"
        print(f"> {BOLD}{directory_to_format}{RESET}")
        try:
            subprocess.run(command_args, check=False, text=True)
        except FileNotFoundError:
            print(f"🔴 Error: 'black' command not found. Please ensure black is installed and in your PATH.")
            break  # Exit if black isn't found
        except Exception as e:
            print(f"🔴 An unexpected error occurred while processing {directory_to_format}: {e}")


if __name__ == "__main__":
    main()
