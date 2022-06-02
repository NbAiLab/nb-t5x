MODEL_DIR="gs://nb-t5/t5/t5_1_1_from_t5_checkpoints"
python -m t5x.train \
  --gin_file=pretrain_t5_1_1_base.gin \
  --gin.MODEL_DIR=\"${MODEL_DIR}\" \
  --alsologtostderr
