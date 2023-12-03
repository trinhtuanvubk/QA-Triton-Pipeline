import numpy as np

from tritonclient.utils import *

# import tritonclient.http as tritonhttpclient
import tritonclient.http
import soundfile
import librosa
from scipy.io.wavfile import write


def main():
    client = tritonclient.http.InferenceServerClient(url="localhost:8050")

    # array, _ = soundfile.read("./sample/recharge.mp3")
    array ,_ = librosa.load("./sample/recharge.mp3")
    array = np.expand_dims(array, axis=0)
    array = np.asarray(array, dtype=np.float32)
    # print(array.shape)
    input_array = tritonclient.http.InferInput("input", array.shape, "FP32")
    input_array.set_data_from_numpy(array)
    output = tritonclient.http.InferRequestedOutput("output")

    response = client.infer(model_name="pipeline", inputs=[input_array], outputs=[output])
    audio = response.as_numpy("output")
    print(audio.shape)
    write(data=np.squeeze(audio[0]), rate=22050, filename="./output_pipeline.wav")


if __name__=="__main__":
    main()