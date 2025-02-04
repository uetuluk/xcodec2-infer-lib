Trying to achieve M chip Macbook support for https://huggingface.co/HKUSTAudio/xcodec2.

# Setting up for dev

conda create xcodec2-infer-lib python=3.12 -y

conda activate xcodec2-infer-lib

pip install -e .

# Setting up for use

pip install xcodec2-infer-lib