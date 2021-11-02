#!/bin/bash

# Update
# Install packages
echo "[INFO] Updating system..."
echo "[INFO] Attempting to install packages listed in apt.txt..."
echo ""
sudo apt -qqq update && xargs -a apt.txt sudo apt -qq install -y
if [[ $? != 0 ]]; then
    echo "[ERR] Update failed, try updating manually."
else
    echo ""
    echo "DONE."
    echo ""
fi

# Check if pip is installed
if [[ $(pip -V > /dev/null 2>&1) ]]; then
    echo "[INFO] Pip is installed, skipping to next step..."
else
    echo "[INFO] Pip is not installed, attempting to install pip..."
    wget https://bootstrap.pypa.io/get-pip.py > /dev/null 2>&1
    python3 get-pip.py > /dev/null 2>&1
    if [[ $? != 0 ]]; then
        echo "[ERR] Installation failed, try installing manually."
    else
        echo ""
        echo "DONE."
        echo ""
    fi
fi

# Install virtualenv with pip
echo "[INFO] Installing virtual environment..."
$HOME/.local/bin/pip3 -qq install virtualenv virtualenvwrapper > /dev/null 2>&1
if [[ $? != 0 ]]; then
    echo "[ERR] Installation failed, try installing manually."
else
    echo ""
    echo "DONE."
    echo ""
fi

# Configure virtualenv and virtualenvwrapper path
if [[ $(which virtualenv) ]]; then
    ve_path=$(which virtualenv)
    vew_path="${ve_path}wrapper.sh"
else
    ve_path="${HOME}/.local/bin/virtualenv"
    vew_path="${HOME}/.local/bin/virtualenvwrapper.sh"
fi

# Create virtualenv strings
ve_a="# virtualenv and virtualenvwrapper"
ve_b="export WORKON_HOME=$HOME/.local/bin/.virtualenvs"
ve_c="export VIRTUALENVWRAPPER_PYTHON=/usr/bin/python3"
ve_d="export VIRTUALENVWRAPPER_VIRTUALENV=$ve_path"
ve_e="source $vew_path"

# Add virtualenv to bash profile
echo "[INFO] Updating bash profile in order to use virtual environment..."
echo "" >> ~/.bashrc
echo "$ve_a" >> ~/.bashrc
echo "$ve_b" >> ~/.bashrc
echo "$ve_c" >> ~/.bashrc
echo "$ve_d" >> ~/.bashrc
echo "$ve_e" >> ~/.bashrc
eval "$(cat ~/.bashrc | tail -n +10)" > /dev/null 2>&1
if [[ $? != 0 ]]; then
    echo "[ERR] Source failed, try sourcing manually."
else
    echo ""
    echo "DONE."
    echo ""
fi

# Create environment
if [[ $(workon detector > /dev/null 2>&1) ]]; then
    echo "[INFO] Virtual environment detector is already created, skipping to next step..."
else
    echo "[INFO] Attempting to create detector virtual enviroment..."
    mkvirtualenv detector -p python3 > /dev/null 2>&1
    if [[ $? != 0 ]]; then
        echo "[ERR] Environment creation failed, try creating manually."
    else
        echo ""
        echo "DONE."
        echo ""
    fi
fi

# Install pip packages from pip.txt
echo "[INFO] Attempting to install packages listed in pip.txt"
$HOME/.local/bin/.virtualenvs/detector/bin/pip -qq install -r pip.txt
if [[ $? != 0 ]]; then
    echo "[ERR] Installation failed, try installing manually."
else
    echo ""
    echo "DONE."
    echo ""
    echo "[INFO] Installation completed."
    echo ""
    echo "The environment detector was successfully created"
    echo "In order to run the python scripts activate the detector virtual environment"
    echo "To do so execute the following command -> # workon detector"
fi
