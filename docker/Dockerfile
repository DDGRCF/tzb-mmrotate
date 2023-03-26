ARG PYTORCH="1.9.0"

ARG CUDA="10.2"
ARG CUDNN="7"

FROM pytorch/pytorch:${PYTORCH}-cuda${CUDA}-cudnn${CUDNN}-devel

ENV TORCH_CUDA_ARCH_LIST="6.0 6.1 7.0 8.6+PTX"
ENV TORCH_NVCC_FLAGS="-Xfatbin -compress-all"
ENV CMAKE_PREFIX_PATH="$(dirname $(which conda))/../"

# To fix GPG key error when running apt-get update
RUN echo "deb http://mirrors.tuna.tsinghua.edu.cn/ubuntu/ xenial main restricted universe multiverse\n \
    deb http://mirrors.tuna.tsinghua.edu.cn/ubuntu/ xenial-updates main restricted universe multiverse\n \
    deb http://mirrors.tuna.tsinghua.edu.cn/ubuntu/ xenial-backports main restricted universe multiverse\n \
    deb http://mirrors.tuna.tsinghua.edu.cn/ubuntu/ xenial-security main restricted universe multiverse\n" > /etc/apt/sources.list

RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub 
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64/7fa2af80.pub

RUN apt-get update && apt-get install -y ffmpeg libsm6 libxext6 git ninja-build libglib2.0-0 libsm6 libxrender-dev libxext6 wget proj-bin cmake\ 
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN mkdir -p ~/.pip && touch ~/.pip/pip.conf \
    && echo "[global]\ntrusted-host =  https://pypi.tuna.tsinghua.edu.cn\nindex-url = https://pypi.tuna.tsinghua.edu.cn/simple" >> ~/.pip/pip.conf

# Install openmim
RUN pip install --no-cache-dir -U openmim
# Install mmengine, mmcv, and mmdetection
RUN mim install --no-cache-dir mmengine "mmcv>=2.0.0rc2" "mmdet>=3.0.0rc2" "mmcls>=1.0.0rc0"
# Install MMRotate
RUN conda clean --all -y
RUN git clone https://gitee.com/open-mmlab/mmrotate.git -b 1.x /mmrotate
WORKDIR /mmrotate
ENV FORCE_CUDA="1"

RUN pip install -r requirements/build.txt
RUN pip install --no-cache-dir -e .
RUN pip install setuptools==57.5.0
# RUN wget https://github.com/OSGeo/gdal/releases/download/v3.0.2/gdal-3.0.2.tar.gz -O gdal.tar.gz \
#   && tar -xf gdal.tar.gz && rm -rf gdal.tar.gz \
#    && cd gdal && ./configure --prefix=/usr/local/gdal && make -j$(nproc) && make install
ARG GDAL="3.0.2"
ARG PROJ="6.3.1"
COPY gdal-${GDAL}.tar.gz /
COPY proj-${PROJ}.tar.gz /

RUN cd / && tar -xf proj-${PROJ}.tar.gz && rm -rf proj-${PROJ}.tar.gz \
    && cd proj-${PROJ} && mkdir -p build && cd build \
    && cmake  -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/usr/local/proj \
    -DENABLE_CURL=OFF -DENABLE_TIFF=OFF -DENABLE_IPO=OFF \
    -DBUILD_LIBPROJ_SHARED=ON \
    -DBUILD_SHARED_LIBS=ON  -DBUILD_TESTING=OFF  -DPROJ_TESTS=OFF \
    -DBUILD_GIE=OFF \
    -DBUILD_CCT=OFF \
    -DBUILD_CS2CS=OFF \
    -DBUILD_GEOD=OFF \
    -DBUILD_PROJINFO=OFF \
    -DBUILD_PROJSYNC=OFF \
    -DBUILD_PROJ=OFF ..\
    && make -j$(nproc) \
    && make install \
    && echo /usr/local/proj/lib >> /etc/ld.so.conf && ldconfig

RUN cd / && tar -xf gdal-${GDAL}.tar.gz && rm -rf gdal-${GDAL}.tar.gz \
    && cd gdal-${GDAL} && ./configure --with-proj=/usr/local/proj --prefix=/usr/local/gdal  \
    && make -j$(nproc) && make install

RUN export CPLUS_INCLUDE_PATH=/usr/local/gdal/include \
    && export C_INCLUDE_PATH=/usr/local/gdal/include \
    && export PATH=/usr/local/gdal/bin:${PATH} \
    && echo /usr/local/gdal/lib >> /etc/ld.so.conf && ldconfig \
    && pip install GDAL==$(gdal-config --version)
