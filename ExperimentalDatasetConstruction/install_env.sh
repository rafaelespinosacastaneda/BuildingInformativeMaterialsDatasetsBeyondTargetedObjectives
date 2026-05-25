#!/bin/bash

set -e

# Required Python major/minor version
required_major_minor="3.11"

# Name of the virtual environment folder
env_name="test"

# Python executable to use
PYTHON_BIN="${PYTHON_BIN:-python3.11}"

# Check that the Python executable exists
if ! command -v "$PYTHON_BIN" &> /dev/null; then
    echo "Error: $PYTHON_BIN was not found."
    echo "Please install Python 3.11 or load the correct module."
    exit 1
fi

# Check Python version
python_version=$("$PYTHON_BIN" -c "import platform; print(platform.python_version())")
python_major_minor=$("$PYTHON_BIN" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")

if [ "$python_major_minor" != "$required_major_minor" ]; then
    echo "Error: Python $required_major_minor.x is required, but found Python $python_version."
    exit 1
fi

# Avoid overwriting an existing environment
if [ -d "$env_name" ]; then
    echo "Error: environment folder '$env_name' already exists."
    echo "Remove it first with:"
    echo "rm -rf $env_name"
    exit 1
fi

# Create a Python virtual environment using Python 3.11.x
"$PYTHON_BIN" -m venv "$env_name"

# Detect activation script and Python executable
if [ -f "$env_name/bin/activate" ]; then
    ACTIVATE_SCRIPT="$env_name/bin/activate"
    VENV_PYTHON="$env_name/bin/python"
elif [ -f "$env_name/Scripts/activate" ]; then
    ACTIVATE_SCRIPT="$env_name/Scripts/activate"
    VENV_PYTHON="$env_name/Scripts/python.exe"
else
    echo "Error: virtual environment was created, but no activation script was found."
    echo "Expected either:"
    echo "$env_name/bin/activate"
    echo "or"
    echo "$env_name/Scripts/activate"
    exit 1
fi

# Activate the environment
source "$ACTIVATE_SCRIPT"

# Confirm the environment Python version
env_python_version=$("$VENV_PYTHON" -c "import platform; print(platform.python_version())")
env_python_major_minor=$("$VENV_PYTHON" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")

if [ "$env_python_major_minor" != "$required_major_minor" ]; then
    echo "Error: Virtual environment uses Python $env_python_version, not Python $required_major_minor.x."
    exit 1
fi

# Upgrade pip
"$VENV_PYTHON" -m pip install --upgrade pip

# Install all required packages
"$VENV_PYTHON" -m pip install -r requirements.txt

echo "Environment '$env_name' was created successfully."
echo "Python version used: $env_python_version"
echo "To activate it, run:"

if [ -f "$env_name/bin/activate" ]; then
    echo "source $env_name/bin/activate"
else
    echo "source $env_name/Scripts/activate"
fi