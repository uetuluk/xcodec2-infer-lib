# xcodec2-infer-lib

CPU support for https://huggingface.co/HKUSTAudio/xcodec2

# Setting up for dev

```
conda create xcodec2-infer-lib python=3.12 -y
```

```
conda activate xcodec2-infer-lib
```

```
pip install -e .
```

# Setting up for use

```
pip install xcodec2-infer-lib
```

# Testing

Replace `your_model_path` with the absolute path to your model.

```
export MODEL_PATH="your_model_path" 
pytest test_xcodec.py
```

# Model

You can download the model from https://huggingface.co/HKUSTAudio/xcodec2.

# Using

Check out the example at https://huggingface.co/HKUSTAudio/Llasa-3B.