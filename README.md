# vehicle-detector
Detection of vehicles using computer vision

# Project Structure

```
|-- vehicle-detector/
|   |-- cnn/
|   |   |-- smallervggnet.py
|   |-- classify.py
|   |-- train.py
```

# Prerequisites

In order to run the Python scripts it is necessary to install certain packages.

It is recommended to create a Python virtual environment to avoid conflicts between packages.

## Automatic Installation

If you want to install all the packages automatically inside a virtual environment you can use the included script `requirements/install.sh`
The script was tested using Ubuntu 20.04 and Ubuntu 21.04. To use the installation script follow the next steps:

> This script is still under development, use with caution.

1. Go to the requirements dir:

```sh
cd requirements
```

2. Make the script executable:

```sh
chmod +x install.sh
```

3. Execute the script using the following command:

```sh
. ./install.sh
```

4. The script will run and it will ask for your password in order to use `apt`.

5. At the end the script will have started the virtual environment automatically, if it does not use the following command:

```sh
workon detector
```

## Manual Installation

Alternatively you can install the requirements in `apt.txt` and `pip.txt` manually. To install the `apt` requirements listed in `apt.txt`:

1. Update your system

```sh
sudo apt update
```

2. Install the packages using the `apt.txt` file:

```sh
xargs -a apt.txt sudo apt install -y
```

To install the `pip` requirements listed in `pip.txt`:

1. Check if pip is installed

```sh
pip -V
```

2. If pip is not installed use the following commands to install it:

```sh
wget https://bootstrap.pypa.io/get-pip.py
```

and

```sh
python3 get-pip.py
```

3. Install `virtualenv` and `virtualenvwrapper` using `pip`:

```sh
pip3 install virtualenv virtualenvwrapper
```

or

```sh
$HOME/.local/bin/pip3 install virtualenv virtualenvwrapper
```

4. Add the next lines to your `bash` profile (`~/.bashrc`) according to your installation paths:

```
# virtualenv and virtualenvwrapper
export WORKON_HOME=$HOME/.local/bin/.virtualenvs
export VIRTUALENVWRAPPER_PYTHON=/usr/bin/python3
export VIRTUALENVWRAPPER_VIRTUALENV=$HOME/.local/bin/virtualenv
source $HOME/.local/bin/virtualenvwrapper.sh
```

or

```
# virtualenv and virtualenvwrapper
export WORKON_HOME=$HOME/.local/bin/.virtualenvs
export VIRTUALENVWRAPPER_PYTHON=/usr/bin/python3
export VIRTUALENVWRAPPER_VIRTUALENV=/usr/local/bin/virtualenv
source /usr/local/bin/virtualenvwrapper.sh
```

5. Update your `bash` profile:

```sh
source ~/.bashrc
```

6. Create the detector virtual environment

```sh
mkvirtualenv detector -p python3
```

7. Install the `pip` requirements listed in `pip.txt`

```sh
$HOME/.local/bin/.virtualenvs/detector/bin/pip install -r pip.txt
```

# Usage

First step is activate the virtual environment

```sh
workon detector
```

Once activated we can run `train.py` or `predict.py`

## Train

Basic usage:

```sh
python train.py -n <test_n> -d <dataset> -id <img_dim> -ts <test_size> -lr <learn_rate> -bs <batch_size> -e <epochs>
```

Where: 

- `-n` or `--number`: Number of test in order to keep control of output files
- `-d` or `--dataset`: Path to input dataset
- `-id` or `--imagedim`: Image dimensions, i.e. 96, 128, etc (for pre-processing)
- `-ts` or `--testsize`: Size of the test partition
- `-lr` or `--learnrate`: Initial learning rate
- `-bs` or `--batchsize`: Batch size
- `-e` or `--epochs`: Number of epochs

Example:

```sh
python train.py -n 1 -d vehicules -id 128 -ts 0.2 -lr 0.001 -bs 32 -e 100
```

## Predict

Basic usage:

```sh
python train.py -td <test_data> -n <test_n> -d <dataset> -id <img_dim> -ts <test_size>
```

Where:

- `-td` or `--testdata`: Path to input dataset for testing
- `-n` or `--number`: Number of test created in training
- `-d` or `--dataset`: Path to input dataset used in training
- `-id` or `--imagedim`: Image dimensions used in training
- `-ts` or `--testsize`: Size of the test partition used in training

Example:

```sh
python predict.py -td ../datasets/vehicules-test -n 1 -d ../datasets/vehicules -id 128 -ts 0.2
```

