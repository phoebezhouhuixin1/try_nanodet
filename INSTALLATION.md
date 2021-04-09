# Install pytorch and torchvision and pycocotools

PyTorch should be installed. TorchVision is also required since we will be using it as our model zoo.
* Ensure that conda is installed: `pip install conda`
* Open Anaconda Prompt and type: `where conda`
* Open environment variables and add in 3 more paths to the "Path" variable (such that it matches `where conda`):  
	C:\Users\Phoebe\Anaconda3\Scripts  
	C:\Users\Phoebe\Anaconda3  
	C:\Users\Phoebe\Anaconda3\Library\bin  
* Reopen command prompt (or vscode, wherever you are running the VideoIO app from), then  
	`conda env create -n myvideoioenvironment -f your\path\to\environment-gpu.yml`    
	`conda activate myvidoeioenvironment`  
* INSTALL THE FOLLOWING IN THE ENV USING CMD TERMINAL, NOT POWERSHELL TERMINAL:  
	`pip install torch===1.7.1+cu110 torchvision===0.8.2+cu110 torchaudio===0.7.2 -f https://download.pytorch.org/whl/torch_stable.html`  
    `conda install git`  
	`pip3 install "git+https://github.com/philferriere/cocoapi.git#egg=pycocotools&subdirectory=PythonAPI"` 
    `conda list`

# Set up NanoDet
* `git clone https://github.com/RangiLyu/nanodet.git`
* `cd nanodet`
* `pip install -r requirements.txt`
* `python setup.py develop`
* `conda install tensorboard==1.15.0` (tensorflow will be upgraded to 1.15.0 as well though we are using pytorch)
* Download Pytorch COCO weights (nanodet-m.pth has input shape 416x416?? or is it 320x320): https://drive.google.com/file/d/1EhMqGozKfqEfw8y9ftbi1jhYu86XoW62/view



