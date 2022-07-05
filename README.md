# nb-t5x

## Code

1. Clone the repo and cd to it
2. Clone https://github.com/google-research/t5x inside with name `t5x_repo` and install in edit mode
3. Symlink `t5x_repo/t5x` to `t5x` in the cloned folder of this repo
3. Install dependencies jax for TPU and seqio (this one from repo)
4. Run `run.sh`

Lists of checkpoints can be found:
- https://console.cloud.google.com/storage/browser/t5-data
- https://console.cloud.google.com/storage/browser/scenic-bucket

If meeting segmentation faults when writing checkpoints to the buckets, the reasone might be `tensorstore` version `0.1.18`. As a temporal fix, try using version `0.1.14` instead. Fixed in newer versions of `tensorstore`. See also https://github.com/google-research/t5x/issues/436 if JAX cannot see the TPUs.

## Vocabs
The folder vocabs contains useful information to create SentencePiece vocabularies.

## Exporting

In order to export models to the Huggingface format, a few things are needed:

1. Use the script https://github.com/huggingface/transformers/blob/main/src/transformers/models/t5/convert_t5x_checkpoint_to_flax.py to to convert the model to Flax. Then load it as `AutoT5ForConditionalGeneration.from_pretrained(..., from_flax=True)` and save it with `.save_pretrained(...)`
2. To convert the vocabulary (if custom), use the `sentencepiece_extractor.py` if BPE from https://github.com/huggingface/tokenizers/tree/main/bindings/python/scripts. If Unigram, follows the `convert` script.
3. Upload to a model repo.
