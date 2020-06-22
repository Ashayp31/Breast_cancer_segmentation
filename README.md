# Breast-Cancer-Detection-Code

## Usage for a GPU lab machine

Clone the repository:

```
cd ~/Projects
git clone https://github.com/Adamouization/Breast-Cancer-Detection-Code
```

Create a repository that will be used to install Tensorflow 2 with CUDA 10 for Python and activate the virtual environment for GPU usage:

```
cd libraries/tf2
tar xvzf tensorflow2-cuda-10-1-e5bd53b3b5e6.tar.gz
sh build.sh
```

Activate the virtual environment:

```
source /Breast-Cancer-Detection-Code/tf2/venv/bin/activate
```

`cd` into the `src` directory and run the code:

```
python3 main.py
```

## Dataset usage

This example will use the [mini-MIAS](http://peipa.essex.ac.uk/info/mias.html) dataset. Create a `data` directory:

```
mkdir data
cd data
```

Download the mini-MIAS dataset: 

```
wget http://peipa.essex.ac.uk/pix/mias/all-mias.tar.gz
```

Unzip the dataset:

```
tar xvzf all-mias.tar.gz
```

