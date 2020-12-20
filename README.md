Dynamic Window Approach
===================================

2D Dynamic Window Approach evaluated on randomly generated environments.


## Requirements

### Python

* Python >= 3.6
* cython
* numpy
* cv2 (Optional for the demo)

### C Libraries (Optional for the Demo)

* SDL
* OpenGL

## Installation

You can directly install Python bindings without compiling the library.

### Compile and Install C Library

```bash
git clone https://github.com/goktug97/DynamicWindowApproach
cd DynamicWindowApproach
mkdir build
cd build
cmake ..
make
sudo make install
# Optional: Build Demo
make demo
```

### Install Python Bindings

#### PyPI

```bash
pip3 install dynamic-window-approach --user
```
#### Source

```bash
git clone https://github.com/Duisterhof/DynamicWindowApproach
cd DynamicWindowApproach
python3 setup.py install --user
```

