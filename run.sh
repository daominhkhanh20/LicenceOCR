pip install kaggle
mkdir models 
mv Model* models
export KAGGLE_USERNAME="daominhkhanh"
export KAGGLE_KEY="aad7a721ee47d7c7dec658896bcefedd"
kaggle datasets init -p models
sed -i 's/INSERT_TITLE_HERE/FbModel2311/g' model/dataset-metadata.json
sed -i 's/INSERT_SLUG_HERE/FbModel2311/g' model/dataset-metadata.json
kaggle datasets create -p ocr --dir-mode zip
