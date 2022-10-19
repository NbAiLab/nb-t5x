MODEL_DIR="gs://nb-t5/t5/t5_1_1_large_from_t5_scandi_unigram"

echo "Retrieving latest checkpoint..."
LAST_CHECKPOINT=$(gsutil ls -l $MODEL_DIR/checkpoint_*/ | sort -k 2 | tail -2 | head -1 | cut -d " " -f 8)
INITIAL_CHECKPOINT=$(python -c "print('$LAST_CHECKPOINT'.rsplit('/', 1)[0])");
echo "Starting from checkpoint $INITIAL_CHECKPOINT"

python -m t5x.train \
  --gin_file=finetune_t5_large_new_scandi_vocab.gin \
  --gin.MODEL_DIR=\"${MODEL_DIR}\" \
  --gin.INITIAL_CHECKPOINT_PATH=\"${INITIAL_CHECKPOINT}\" \
  --alsologtostderr

