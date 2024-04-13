wget https://zenodo.org/record/1203745/files/UrbanSound8K.tar.gz -O urban8k.tgz
tar -xzf urban8k.tgz
rm urban8k.tgz
python -m venv .env
source .env/bin/activate
pip install -r requirements.txt
python data_augmentation.py
python extract.py
