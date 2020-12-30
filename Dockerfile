FROM tensorflow/tensorflow:2.3.1-gpu

# ensure local python is preferred over distribution python
ENV PATH /usr/local/bin:$PATH

# http://bugs.python.org/issue19846
# > At the moment, setting "LANG=C" on a Linux system *fundamentally breaks Python 3*, and that's not OK.
ENV LANG C.UTF-8

# Install python tools and dev packages
RUN apt-get update \
    && apt-get install -q -y --no-install-recommends python3.7-dev python3-pip python3-setuptools python3-wheel gcc \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*
RUN update-alternatives --install /usr/bin/python3 python /usr/bin/python3.7 1
RUN easy_install pip
RUN pip install --upgrade pip

# Base settings
RUN pip install tensorflow==2.3.1 Keras==2.4.0
RUN apt-get update
RUN apt-get install 'ffmpeg'\
    'libsm6'\ 
    'libxext6' -y
RUN pip install opencv-python
RUN pip install common-py image-keras==0.3.4
