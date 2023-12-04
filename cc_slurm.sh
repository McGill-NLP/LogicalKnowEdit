module load StdEnv/2020 
module load python/3.9
module load gcc/9.3.0 cuda/11.0
module load arrow/9.0.0
cd $SLURM_TMPDIR
virtualenv --no-download venv
source $SLURM_TMPDIR/venv/bin/activate
module load arrow/9.0.0
pip3 install torch --no-index
pip install numpy==1.24.2 --no-index
pip3 install scikit-learn==1.0.1 --no-index
pip3 install tqdm --no-index
pip3 install transformers==4.33.3 --no-index
pip3 install datasets --no-index
pip3 install sentencepiece --no-index
pip3 install protobuf --no-index
pip3 install pytorch_lightning==1.2.1 --no-index 
pip3 install omegaconf --no-index
pip3 install hydra-core --no-index
cd /home/lcc/knowedit/higher/ && python3 setup.py install