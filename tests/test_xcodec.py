import os
import pytest
import numpy as np
import wave
from safetensors.torch import load_file
from xcodec2.modeling_xcodec2 import XCodec2Model

MODEL_PATH = os.getenv("MODEL_PATH")
TOKEN_PATH_SAFETENSORS = os.path.join(os.path.dirname(__file__), "speech_tokens.safetensors")


@pytest.fixture(scope="module")
def codec_model():
    model = XCodec2Model.from_pretrained(MODEL_PATH)
    model.eval()
    model.to("cpu")
    return model

def test_model_loading(codec_model):
    assert codec_model is not None

def test_safetensors_loading():
    loaded_data = load_file(TOKEN_PATH_SAFETENSORS, device="cpu")
    assert "speech_tokens" in loaded_data
    assert loaded_data["speech_tokens"].ndim > 0

def test_decoding(codec_model):
    loaded_data = load_file(TOKEN_PATH_SAFETENSORS, device="cpu")
    speech_tokens = loaded_data["speech_tokens"]
    gen_wav = codec_model.decode_code(speech_tokens)
    assert gen_wav is not None
    assert gen_wav.ndim == 3

def test_wav_file_generation(codec_model):
    loaded_data = load_file(TOKEN_PATH_SAFETENSORS, device="cpu")
    speech_tokens = loaded_data["speech_tokens"]
    gen_wav = codec_model.decode_code(speech_tokens)
    wav_data = gen_wav[0, 0, :].cpu().numpy()

    wav_data = np.int16(wav_data / np.max(np.abs(wav_data)) * 32767)

    output_path = "test.wav"
    with wave.open(output_path, "w") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(16000)
        wav_file.writeframes(wav_data.tobytes())

    assert os.path.exists(output_path)
    assert os.path.getsize(output_path) > 0

    os.remove(output_path)
