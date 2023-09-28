import numpy as np

from tritonclient.utils import *

# import tritonclient.http as tritonhttpclient
import tritonclient.http
import soundfile
import librosa

def main():
    client = tritonclient.http.InferenceServerClient(url="localhost:8050")

    # array, _ = soundfile.read("./sample/recharge.mp3")
    array ,_ = librosa.load("./sample/recharge.mp3")
    array = np.expand_dims(array, axis=0)
    array = np.asarray(array, dtype=np.float32)
    print(array.shape)
    input_array = tritonclient.http.InferInput("input", array.shape, "FP32")

    input_array.set_data_from_numpy(array)

    output = tritonclient.http.InferRequestedOutput("paraphrase_answer")

    response = client.infer(model_name="pipeline", inputs=[input_array], outputs=[output]
                            )

    print(response.as_numpy("paraphrase_answer")) 


if __name__=="__main__":
    main()