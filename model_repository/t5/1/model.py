import numpy as np
import sys
import os
import json
from pathlib import Path

import torch
import triton_python_backend_utils as pb_utils
from transformers import T5ForConditionalGeneration, T5Tokenizer

T5_CONFIG = "config"
T5_PATH = "checkpoints"
cur_folder = Path(__file__).parent

t5_config = str(cur_folder/T5_CONFIG)
t5_path = str(cur_folder/T5_PATH)

class TritonPythonModel:
    def initialize(self, args):

        self.output_dtype = pb_utils.triton_string_to_numpy(
            pb_utils.get_output_config_by_name(
                json.loads(args['model_config']), "output_ids"
            )['data_type']
        )

        self.model = T5ForConditionalGeneration.from_pretrained(t5_path, local_files_only=True).cuda()
        print("TritonPythonModel initialized")

    def execute(self, requests):
        responses = []
        for request in requests:
            input_ids = pb_utils.get_input_tensor_by_name(request, "input_ids")
            input_ids = input_ids.as_numpy()
            input_ids = torch.as_tensor(input_ids).long().cuda()
            attention_mask = pb_utils.get_input_tensor_by_name(request, "attention_mask")
            attention_mask = attention_mask.as_numpy()
            attention_mask = torch.as_tensor(attention_mask).long().cuda()
            inputs = {'input_ids': input_ids, 'attention_mask': attention_mask}
            translation = self.model.generate(**inputs,
                                              max_length=256,
                                              num_beams=10,
                                              num_return_sequences=10,
                                              temperature=1.3,
                                              do_sample=True)
            # Convert to numpy array on cpu:
            np_translation =  translation.cpu().int().detach().numpy()
            inference_response = pb_utils.InferenceResponse(
                output_tensors=[
                    pb_utils.Tensor(
                        "output_ids",
                        np_translation.astype(self.output_dtype)
                    )
                ]
            )
            responses.append(inference_response)
        return responses

    def finalize(self):
        """`finalize` is called only once when the model is being unloaded.
        Implementing `finalize` function is optional. This function allows
        the model to perform any necessary clean ups before exit.
        """
        print('Cleaning up...')