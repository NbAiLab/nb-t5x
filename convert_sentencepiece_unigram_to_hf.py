import shutil
import argparse
import os
import tempfile
from pathlib import Path

from transformers import T5Tokenizer


def main(bucket_tokenizer_path, local_path):
    """
    mv ./no-da-en-sv-nn-is_32000_unigram.sp.model ./spiece.model
wget "https://raw.githubusercontent.com/huggingface/tokenizers/6887c0f04d43cb2f26f8c93aeacc8b0ee3e1b570/bindings/python/scripts/sentencepiece_extractor.py" -O sentencepiece_extractor.py
python sentencepiece_extractor.py --provider sentencepiece --model ./spiece.model --vocab-output-path ./vocab.json --merges-output-path ./merges.txt
python -c "from transformers import T5Tokenizer; t='Jeg leser boken'; print(t, T5Tokenizer.from_pretrained('./').encode(t))"
"""
    tokenizer_path = Path(bucket_tokenizer_path)
    tmp = Path(tempfile.gettempdir())
    local = Path(local_path)
    cmd = f"gsutil -m cp -r {bucket_tokenizer_path} {tmp}"
    os.system(cmd)
    cmd = f"mv {tmp/tokenizer_path.name} {local/'spiece.model'}"
    os.system(cmd)
    cmd = f"python sentencepiece_extractor.py --provider sentencepiece --model {local/'spiece.model'} --vocab-output-path {local/'vocab.json'} --merges-output-path {local/'merge.txt'}"
    os.system(cmd)
    text = "Jeg leser boken"
    print(text, T5Tokenizer.from_pretrained(local).tokenize(text))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument(
        "--bucket_tokenizer_path", default=None, type=str, required=True, help="Path to SentencePiece Unigram tokenizer in the bucket."
    )
    parser.add_argument(
        "--local_path", default=None, type=str, required=True, help="Path to local directory."
    )
    args = parser.parse_args()
    main(args.bucket_tokenizer_path, args.local_path)
