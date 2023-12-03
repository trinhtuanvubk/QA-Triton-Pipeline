FROM nvcr.io/nvidia/tritonserver:21.10-py3
# RUN apt-key del 7fa2af80
# RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/3bf863cc.pub
# RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu2004/x86_64/7fa2af80.pub
# # RUN apt-get update
WORKDIR /workspace
ENV CUDA_HOME=/usr/local/cuda
ENV TZ=Europe/London
ENV HOME=/config
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64/7fa2af80.pub
RUN apt update -q
RUN apt install espeak -y
RUN pip install --upgrade pip
RUN pip install torch==2.0.1 transformers pathlib loguru pandas openpyxl \
    python-Levenshtein==0.12.2 six==1.16.0 ctc-segmentation ctc-segmentation==1.7.1 \
    Unidecode==1.3.6 phonemizer==3.2.1
RUN git clone --recursive https://github.com/parlance/ctcdecode.git \
    && cd ctcdecode && pip install .
