#!/usr/bin/env python3
"""
Virtual environment runner module - handles venv creation and execution.
"""

import os
import sys
import subprocess
import venv
from pathlib import Path
from typing import List, Optional

class VenvRunner:
    """Manages virtual environment and runs commands within it."""

    def __init__(self, script_dir: Optional[Path] = None):
        """Initialize the VenvRunner.

        Args:
            script_dir: Directory containing the script. Defaults to current directory.
        """
        self.script_dir = script_dir or Path.cwd()
        self.venv_dir = self.script_dir / "venv"
        self.requirements_file = self.script_dir / "requirements.txt"

    def get_venv_python(self) -> Path:
        """Get the path to the Python executable in the virtual environment."""
        if sys.platform == "win32":
            return self.venv_dir / "Scripts" / "python.exe"
        else:
            return self.venv_dir / "bin" / "python"

    def get_venv_pip(self) -> Path:
        """Get the path to the pip executable in the virtual environment."""
        if sys.platform == "win32":
            return self.venv_dir / "Scripts" / "pip.exe"
        else:
            return self.venv_dir / "bin" / "pip"

    def create_venv(self) -> None:
        """Create virtual environment if it doesn't exist."""
        if not self.venv_dir.exists():
            print("Creating virtual environment...")
            venv.create(self.venv_dir, with_pip=True)
            print("Virtual environment created.")

    def install_requirements(self) -> None:
        """Install requirements if requirements.txt exists."""
        if self.requirements_file.exists():
            print("Installing requirements...")
            pip_path = self.get_venv_pip()
            subprocess.check_call(
                [str(pip_path), "install", "-q", "-r", str(self.requirements_file)],
                stdout=subprocess.DEVNULL
            )
            print("Requirements installed.")

    def check_requirements(self, required_packages: List[str]) -> bool:
        """Check if required packages are installed.

        Args:
            required_packages: List of package names to check.

        Returns:
            True if all packages are installed, False otherwise.
        """
        pip_path = self.get_venv_pip()
        result = subprocess.run(
            [str(pip_path), "freeze"],
            capture_output=True,
            text=True
        )
        installed = result.stdout.lower()

        for package in required_packages:
            if package.lower() not in installed:
                return False
        return True

    def ensure_venv(self, required_packages: Optional[List[str]] = None) -> None:
        """Ensure virtual environment exists and has required packages.

        Args:
            required_packages: List of package names to check. If None, checks requirements.txt.
        """
        # Create venv if it doesn't exist
        if not self.venv_dir.exists():
            self.create_venv()
            self.install_requirements()
        elif required_packages and not self.check_requirements(required_packages):
            print("Missing dependencies detected.")
            self.install_requirements()
        elif not required_packages and self.requirements_file.exists():
            # Check if we need to install/update requirements
            # For simplicity, we'll check if key packages are installed
            pip_path = self.get_venv_pip()
            result = subprocess.run(
                [str(pip_path), "freeze"],
                capture_output=True,
                text=True
            )
            if not result.stdout.strip():
                # No packages installed
                self.install_requirements()

    def run_python_module(self, module: str, args: List[str]) -> int:
        """Run a Python module in the virtual environment.

        Args:
            module: Module name to run (e.g., 'pytest').
            args: Arguments to pass to the module.

        Returns:
            Exit code from the module execution.
        """
        self.ensure_venv()
        python_path = self.get_venv_python()
        cmd = [str(python_path), "-m", module] + args

        try:
            result = subprocess.run(cmd)
            return result.returncode
        except KeyboardInterrupt:
            return 1

    def run_python_script(self, script: Path, args: List[str]) -> int:
        """Run a Python script in the virtual environment.

        Args:
            script: Path to the Python script.
            args: Arguments to pass to the script.

        Returns:
            Exit code from the script execution.
        """
        self.ensure_venv()
        python_path = self.get_venv_python()
        cmd = [str(python_path), str(script)] + args

        # Set PYTHONPATH to include the project root
        env = os.environ.copy()
        env['PYTHONPATH'] = str(self.script_dir)

        try:
            result = subprocess.run(cmd, env=env)
            return result.returncode
        except KeyboardInterrupt:
            return 1

    def run_command(self, cmd: List[str]) -> int:
        """Run a command in the virtual environment.

        Args:
            cmd: Command and arguments to run.

        Returns:
            Exit code from the command execution.
        """
        self.ensure_venv()

        # Prepend venv bin directory to PATH
        env = os.environ.copy()
        if sys.platform == "win32":
            venv_bin = self.venv_dir / "Scripts"
        else:
            venv_bin = self.venv_dir / "bin"
        env["PATH"] = str(venv_bin) + os.pathsep + env.get("PATH", "")

        try:
            result = subprocess.run(cmd, env=env)
            return result.returncode
        except KeyboardInterrupt:
            return 1


def run_in_venv(script_name: str, script_dir: Optional[Path] = None, args: Optional[List[str]] = None) -> None:
    """Convenience function to run a script in venv.

    Args:
        script_name: Name of the script to run (without .py extension).
        script_dir: Directory containing the script. Defaults to current directory.
        args: Arguments to pass to the script. Defaults to sys.argv[1:].
    """
    runner = VenvRunner(script_dir)
    script_path = runner.script_dir / f"{script_name}.py"
    exit_code = runner.run_python_script(script_path, args or sys.argv[1:])
    sys.exit(exit_code)


def run_module_in_venv(module: str, script_dir: Optional[Path] = None, args: Optional[List[str]] = None) -> None:
    """Convenience function to run a module in venv.

    Args:
        module: Module name to run (e.g., 'pytest').
        script_dir: Directory for the venv. Defaults to current directory.
        args: Arguments to pass to the module. Defaults to sys.argv[1:].
    """
    runner = VenvRunner(script_dir)
    exit_code = runner.run_python_module(module, args or sys.argv[1:])
    sys.exit(exit_code)