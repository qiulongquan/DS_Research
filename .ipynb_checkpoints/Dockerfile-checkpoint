#################################################
# Data Science Base Dockerfile 
# 
# Base Docker image: tensorflow/tensorflow:latest-gpu-jupyter
# User: root (for simplicity)
# 这个是最新版的Docker file（2020-11-08），是现在使用的版本，更好用。加入了anaconda，jupyter notebook，jupyter lab等package
#################################################

# Base docker image
# -devel 这个tags是装载Bazel TensorFlow开发组件
FROM tensorflow/tensorflow:2.0.3-gpu-py3-jupyter
# FROM tensorflow/tensorflow:latest-gpu-jupyter
# FROM tensorflow/tensorflow:latest-gpu-jupyter

# Timezone and Maintainer info
ENV TZ Asia/Tokyo
MAINTAINER Qiulongquan

# Installing developer tools and video/image display optimizations
RUN apt-get update -yq
RUN apt-get install lsof -yq
RUN apt-get install wget -yq

RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-4.5.11-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh && \
    /opt/conda/bin/conda clean -tipsy && \
    ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
    echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc && \
    echo "conda activate base" >> ~/.bashrc


# Expose ports 22: ssh,  jupyter nb: 8080, django: 8000
EXPOSE 22
EXPOSE 80
EXPOSE 8000
EXPOSE 8080
EXPOSE 8888

# Changing work directory to workspace
WORKDIR /workspace
# COPY requirements.txt /workspace/
RUN pip install --upgrade pip
# RUN pip install -r requirements.txt

COPY . /workspace/
# using CMD command to run, page contents will automatic show on the web page
# CMD python3 /workspace/manage.py runserver 0.0.0.0:8000

# Commands 
CMD [ "/bin/bash" ]