import numpy as np

from tritonclient.utils import *

# import tritonclient.http as tritonhttpclient
import tritonclient.http
import soundfile
import librosa

def wav2vec():
    client = tritonclient.http.InferenceServerClient(url="0.0.0.0:8050")

    array, _ = soundfile.read("./sample/test.wav")
    array = np.expand_dims(array, axis=0)
    array = np.asarray(array, dtype=np.float32)
    print(array.shape)
    input_array = tritonclient.http.InferInput("input", (1, array.shape[1]), "FP32")

    input_array.set_data_from_numpy(array)

    output = tritonclient.http.InferRequestedOutput("output")

    response = client.infer(model_name="wav2vec2", inputs=[input_array], outputs=[output]
                            )

    print(response.as_numpy("output")) 

if __name__=="__main__":
    # main()
    wav2vec()