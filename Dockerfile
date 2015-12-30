FROM nvidia/cuda:7.0-devel

RUN apt-get update && apt-get install -y python-dev
RUN apt-get install -y python-pip
RUN apt-get install -y python-numpy
RUN apt-get install -y python-scipy
RUN apt-get install -y protobuf-compiler
# For matplotlib
RUN apt-get install -y python-matplotlib
# RUN apt-get install -y libpng12-dev libfreetype6-dev libxft-dev
RUN apt-get install -y libprotobuf-dev


RUN pip install protobuf
RUN pip install nearpy
RUN pip install lmdb


RUN apt-get install -y cmake cmake-curses-gui
RUN apt-get install -y libopenblas-dev
RUN apt-get install -y libboost-python-dev


# Replace 1000 with your user / group id
RUN export uid=1000 gid=1000 && \
    mkdir -p /home/developer && \
    echo "developer:x:${uid}:${gid}:Developer,,,:/home/developer:/bin/bash" >> /etc/passwd && \
    echo "developer:x:${uid}:" >> /etc/group && \
    echo "developer ALL=(ALL) NOPASSWD: ALL" > /etc/sudoers.d/developer && \
    chmod 0440 /etc/sudoers.d/developer && \
    chown ${uid}:${gid} -R /home/developer

USER developer
ENV HOME /home/developer

CMD python setup.py test