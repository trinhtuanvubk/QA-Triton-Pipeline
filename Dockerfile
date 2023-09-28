FROM nvcr.io/nvidia/tritonserver:21.10-py3
RUN pip install --upgrade pip
RUN pip install torch transformers pathlib loguru pandas openpyxl

