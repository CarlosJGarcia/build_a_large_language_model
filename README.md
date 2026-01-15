**Instalación del entorno** \
conda create -n llm_scratch_cuda python=3.11 -y \
conda activate llm_scratch_cuda \
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia \
conda install -c conda-forge matplotlib pandas tqdm jupyterlab \
conda install -c conda-forge tiktoken 

**Importante:** \ 
No hacer $ conda update --all -c conda-forge -y porque instala una versión de PyTorch sin CUDA \ 
Para comprobar que está funcionando CUDA en PyTorch ejecutar /appendix_a/version.py

