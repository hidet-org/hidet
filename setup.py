import os
import shutil
import subprocess
from setuptools import setup
from setuptools.command.build_py import build_py


class CustomBuildCommand(build_py):
    """
    Custom build command to:
    1. Compile the C++/CMake library.
    2. Copy the compiled `.so` files to `python/hidet/lib/`.
    3. Copy the headers to `python/hidet/include/`.
    """

    def run(self):
        repo_root = os.path.abspath(os.path.dirname(__file__))
        build_dir = os.path.join(repo_root, "build")
        lib_src_dir = os.path.join(build_dir, "lib")
        lib_dest_dir = os.path.join(repo_root, "python", "hidet", "lib")
        include_src_dir = os.path.join(repo_root, "include")
        include_dest_dir = os.path.join(repo_root, "python", "hidet", "include")

        # Step 1: Build the library using CMake
        if not os.path.exists(build_dir):
            os.makedirs(build_dir)

        try:
            subprocess.run(["cmake", ".."], cwd=build_dir, check=True)
            subprocess.run(["make", "-j8"], cwd=build_dir, check=True)
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Build failed: {e}")

        # Step 2: Copy generated shared libraries to python/hidet/lib/
        if not os.path.exists(lib_dest_dir):
            os.makedirs(lib_dest_dir)

        for lib_file in ["libhidet_runtime.so", "libhidet.so"]:
            src_path = os.path.join(lib_src_dir, lib_file)
            dest_path = os.path.join(lib_dest_dir, lib_file)
            if os.path.exists(src_path):
                shutil.copy(src_path, dest_path)
                print(f"Copied {lib_file} to {lib_dest_dir}")
            else:
                raise FileNotFoundError(f"Expected library file '{src_path}' not found.")

        # Step 3: Copy headers to python/hidet/include/
        if not os.path.exists(include_dest_dir):
            os.makedirs(include_dest_dir)

        for root, _, files in os.walk(include_src_dir):
            rel_path = os.path.relpath(root, include_src_dir)
            dest_path = os.path.join(include_dest_dir, rel_path)
            os.makedirs(dest_path, exist_ok=True)

            for file in files:
                src_file = os.path.join(root, file)
                dest_file = os.path.join(dest_path, file)
                shutil.copy(src_file, dest_file)
                print(f"Copied {file} to {dest_path}")

        # Continue with normal build process
        super().run()


setup(
    cmdclass={"build_py": CustomBuildCommand},
)