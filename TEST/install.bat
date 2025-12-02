echo This script requires Python 3.14 and CUDA 13.0
python.exe --version
python.exe -m pip install --upgrade pip
pause
pip uninstall -y diffusers torch torchvision
pip install torch==2.9.0 torchvision==0.24.0+cu130 torchaudio==2.9.0 --index-url https://download.pytorch.org/whl/cu130
pip install diffusers accelerate optimum-quanto protobuf sentencepiece transformers pandas huggingface_hub
pip install --upgrade torchao
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0)}')"
pause
