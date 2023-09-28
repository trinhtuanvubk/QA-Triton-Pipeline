from transformers import RobertaTokenizer

import tritongrpcclient as grpcclient
import numpy as np
from tritonclient.utils import *


client = grpcclient.InferenceServerClient('0.0.0.0:8051')


tokenizer = RobertaTokenizer.from_pretrained('roberta-base', local_files_only=True)

jointbert_inputs = tokenizer(["customer service"], padding=True, truncation=True, return_tensors='np')

_input_ids = jointbert_inputs['input_ids']
_attention_mask = jointbert_inputs['attention_mask']

input_ids = grpcclient.InferInput('input_ids', shape=_input_ids.shape, datatype=np_to_triton_dtype(_input_ids.dtype)) 
attention_mask = grpcclient.InferInput('attention_mask', _attention_mask.shape, datatype=np_to_triton_dtype(_attention_mask.dtype))

input_ids.set_data_from_numpy(_input_ids)
attention_mask.set_data_from_numpy(_attention_mask)

out = grpcclient.InferRequestedOutput("slot_logits")

res = client.infer(model_name='jointbert', inputs=[input_ids, attention_mask], outputs=[out])

out_tokens = res.as_numpy("slot_logits").argmax(axis=-1)
print(out_tokens)

