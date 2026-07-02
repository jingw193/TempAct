from setuptools import setup, find_packages

setup(
    name="tempact",
    version="0.0.1",
    packages=find_packages(),
    python_requires=">=3.12",
    install_requires=[
        # torch / torchvision / triton are installed separately from the
        # PyTorch index (see README); pin lower bounds only here.
        "torch>=2.7.1",
        "torchvision>=0.22.1",
        "transformers==4.56.1",
        "accelerate==1.10.1",
        "diffusers==0.37.1",
        "peft==0.17.1",
        "safetensors==0.6.2",
        "tokenizers==0.21.4",
        "huggingface-hub==0.34.4",
        "datasets==4.0.0",
        "sentencepiece==0.2.1",
        "bitsandbytes==0.46.1",
        "einops==0.8.1",

        "numpy==2.4.4",
        "pandas==2.3.2",
        "scipy==1.16.2",
        "scikit-learn==1.7.2",

        "opencv-python==4.13.0.92",
        "pillow==11.2.1",
        "imageio==2.37.3",
        "imageio-ffmpeg==0.6.0",
        "av==17.0.1",
        "decord==0.6.0",
        "mediapy==1.1.6",
        "matplotlib==3.10.6",

        "openai>=1.0",
        "qwen-vl-utils>=0.0.14",

        "omegaconf==2.3.0",
        "ml_collections==1.1.0",
        "easydict==1.13",
        "absl-py==2.3.0",
        "ftfy==6.3.1",
        "pydantic==2.11.7",
        "requests==2.32.4",
        "aiohttp==3.12.15",
        "tqdm==4.67.1",
        "wandb==0.21.4",
        "nvidia-ml-py==13.580.82",

        "fastapi==0.116.1",
        "uvicorn==0.35.0",
    ],
    extras_require={
        "dev": [
            "ipython",
            "pytest",
        ]
    }
)
