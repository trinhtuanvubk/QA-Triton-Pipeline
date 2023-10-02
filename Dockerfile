FROM nvcr.io/nvidia/tritonserver:21.10-py3
RUN pip install --upgrade pip
RUN pip install torch transformers pathlib loguru pandas openpyxl python-Levenshtein==0.12.2 six==1.16.0 ctc-segmentation ctc-segmentation==1.7.1
RUN git clone --recursive https://github.com/parlance/ctcdecode.git \
  && cd ctcdecode && pip install .
