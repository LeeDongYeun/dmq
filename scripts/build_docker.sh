#!/bin/bash

rm -rf Dockerfile
echo "FROM ledoye/imaginaire:24.04-py3" > Dockerfile
input="Dockerfile.base"

echo "RUN python -m pip install --upgrade pip" >> Dockerfile
echo "RUN pip install diffusers==0.30.3 omegaconf pytorch-lightning==1.4.2 torchmetrics==0.6.0 kornia==0.6.0" >> Dockerfile
echo "RUN pip install tensorflow==2.17 tensorflow[and-cuda]" >> Dockerfile
echo "RUN pip install -e git+https://github.com/CompVis/taming-transformers.git@master#egg=taming-transformers" >> Dockerfile
echo "RUN pip install -e git+https://github.com/openai/CLIP.git@main#egg=clip" >> Dockerfile
echo "RUN pip install -e git+https://github.com/CompVis/stable-diffusion.git#egg=latent-diffusion" >> Dockerfile

docker build -t dmq:0.0 .