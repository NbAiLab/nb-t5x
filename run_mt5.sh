MODEL_DIR="gs://nb-t5/t5/mt5_base"

python -m t5x.train \
  --gin_file=finetune_mt5_base.gin \
  --gin.MODEL_DIR=\"${MODEL_DIR}\" \
  --alsologtostderr

