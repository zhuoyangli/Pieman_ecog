# Pieman_ecog

Encoding and decoding models for ECoG data collected from participants listening to Pieman story

# Setup
## setup up virtual env
conda create -n enc python=3.9
conda activate enc

## install dependencies
pip install -r requirements.txt

## install pre-commit
pre-commit install

## run pre-commit against all files once
pre-commit run --all-files

