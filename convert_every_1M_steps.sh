#!/usr/bin/env bash

size="large"
hf_user="versae"
mkdir checkpoints
cd checkpoints
for i in {2..10}; do
    yes |huggingface-cli repo create "t5-${i}m-${size}"
    git clone "https://huggingface.co/${hf_user}/t5-${i}m-${size}"
    cd "t5-${i}m-${size}"
    git lfs install
    huggingface-cli lfs-enable-largefiles .

    python ../nb-t5x/convert_t5x_checkpoint_to_flax.py --t5x_checkpoint_path "gs://nb-t5/t5/t5_1_1_${size}_from_t5_scandi_unigram/checkpoint_${i}000000/" --flax_dump_folder_path ./ --config_name "google/t5-v1_1-${size}"
    python ../nb-t5x/convert_t5x_checkpoint_to_pytorch.py --t5x_checkpoint_path "gs://nb-t5/t5/t5_1_1_${size}_from_t5_scandi_unigram/checkpoint_${i}000000/" --pytorch_dump_path ./ --config_file ./config.json
    python -c "from transformers import T5Tokenizer; T5Tokenizer.from_pretrained('NbAiLab/nb-t5-base').save_pretrained('./')"

    git add -A
    git status
    git commit -m "${i}M ${size}"
    git push 
    cd ..
done
cd ..
