import numpy as np

from tritonclient.utils import *

# import tritonclient.http as tritonhttpclient
import tritonclient.http
import soundfile
import librosa
from transformers import T5ForConditionalGeneration, AutoTokenizer


t5_model = T5ForConditionalGeneration.from_pretrained("./model_repository/t5/1/checkpoints", local_files_only=True).to('cuda')
tokenizer = AutoTokenizer.from_pretrained("./model_repository/t5/1/config", local_files_only=True)
def t5_client():
    client = tritonclient.http.InferenceServerClient(url="0.0.0.0:8050")
    answers = ["I love you so much", "I want to go to school"]
    batch = tokenizer(['Paraphrasing this sentence: ' + answer for answer in answers],
                    truncation=True,
                    padding='longest',
                    max_length=256,
                    return_tensors="np")
    # print(array.shape)
    _input_ids = np.asarray(batch['input_ids'], dtype=np.int32)
    _attention_mask = np.asarray(batch['attention_mask'], dtype=np.int32)
    input_ids = tritonclient.http.InferInput("input_ids", _input_ids.shape, datatype=np_to_triton_dtype(_input_ids.dtype))
    attention_mask = tritonclient.http.InferInput("attention_mask", _attention_mask.shape, datatype=np_to_triton_dtype(_attention_mask.dtype))

    input_ids.set_data_from_numpy(_input_ids)
    attention_mask.set_data_from_numpy(_attention_mask)

    output = tritonclient.http.InferRequestedOutput("output_ids")

    response = client.infer(model_name="t5", inputs=[input_ids, attention_mask], outputs=[output]
                            )

    output_ids = response.as_numpy("output_ids")
    print(output_ids.shape)
    print(output_ids)
    num_return_sequence=10
    all_tgt_texts = [tokenizer.batch_decode(output_ids[i*num_return_sequence:(i+1)*num_return_sequence], skip_special_tokens=True) for i in range(2)]
    print(all_tgt_texts)
    import random
    final_answers = [random.choice(answer) for answer in all_tgt_texts]
    print(final_answers)
if __name__=="__main__":
    # main()
    t5_client()