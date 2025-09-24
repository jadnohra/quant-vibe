# Accel-Vibe-Code

Python project for accelerometer vibration analysis with automated virtual environment management.

## Quick Start

1. **Initial setup** (run once):
   ```bash
   ./setup.sh
   ```

2. **Run Python scripts**:
   ```bash
   ./shell python main.py
   ```

3. **Run tests**:
   ```bash
   ./test
   ```

## Project Structure

- `venv_runner.py` - Virtual environment management module
- `setup.sh` - Initial setup script
- `shell` - Run commands within the venv
- `test` - Run pytest within the venv
- `main.py` - Example vibration analysis script
- `requirements.txt` - Python dependencies

## Usage Examples

```bash
# Run any Python script
./shell python main.py

# Run Python module
./shell python -m pip list

# Run interactive Python
./shell python

# Run specific test file
./test tests/test_main.py

# Run with pytest options
./test -v --cov
```

## Features

- Automatic venv creation and dependency installation
- Isolated Python environment
- Example vibration signal generation and analysis
- FFT frequency spectrum analysis
- Statistical analysis of signals