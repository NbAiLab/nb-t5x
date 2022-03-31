MODEL_DIR="/data/pretrain-model-t5x/"
python -m t5x.train \
  --gin_file=pretrain_t5_1_1_base.gin \
  --gin.MODEL_DIR=\"${MODEL_DIR}\" \
  --alsologtostderr
