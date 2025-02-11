import subprocess
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict

# Configuration
DIRECTORIES2LINELENGTH = {
    "../../python/hidet": 120,
    "../../tests": 120,
    "../../gallery": 100
}
BLACK_ARGS = [
    "python", "-m", "black",
    "--verbose",
    "--skip-string-normalization",
    "--skip-magic-trailing-comma",
]


def main():
    # Collect all Python files
    py_files = {}
    for dir_path, line_length in DIRECTORIES2LINELENGTH.items():
        path = Path(dir_path)
        if path.exists() and path.is_dir():
            for py_file in [str(p) for p in path.rglob("*.py")]:
                py_files[py_file] = line_length

    # print(f"Found {len(py_files)} Python files to format")

    # Track results
    results = defaultdict(list)

    # Process files with progress bar
    with tqdm(total=len(py_files), desc="Formatting", ncols=80) as pbar:
        for file_path, line_length in py_files.items():
            try:
                # Run black and capture output
                result = subprocess.run(
                    [*BLACK_ARGS, "--line-length", str(line_length), file_path],
                    check=True,
                    capture_output=True,
                    text=True
                )

                # Parse black output
                for line in result.stderr.splitlines():
                    if 'reformatted' in line:
                        results['formatted'].append(file_path)
                        tqdm.write(f"🟢 Reformatted: {file_path}")
                    elif 'already well formatted' in line or "wasn't modified on disk since last run" in line:
                        results['unchanged'].append(file_path)
                        # tqdm.write(f"🟡 Unchanged:    {file_path}")

            except subprocess.CalledProcessError as e:
                results['errors'].append((file_path, e.stderr))
                tqdm.write(f"🔴 Error:        {file_path}\n{'-' * 50}\n{e.stderr}\n{'-' * 50}")

            pbar.update(1)

    # Print summary
    # print("\nFormatting Summary:")
    # print(f"🟢 Reformatted:  {len(results['formatted'])} files")
    # print(f"🟡 Unchanged:    {len(results['unchanged'])} files")
    # print(f"🔴 Errors:       {len(results['errors'])} files")
    #
    # # Show example files in each category
    # def show_examples(category, emoji, max_examples=5):
    #     if results[category]:
    #         print(f"\n{emoji} {category.capitalize()} files:")
    #         for f in results[category][:max_examples]:
    #             print(f"  - {f}")
    #         if len(results[category]) > max_examples:
    #             print(f"  ... and {len(results[category]) - max_examples} more")
    #
    # show_examples('formatted', '🟢')
    # show_examples('unchanged', '🟡')
    # show_examples('errors', '🔴')


if __name__ == "__main__":
    main()