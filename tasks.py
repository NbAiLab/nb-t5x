import functools

import seqio
import tensorflow as tf
import t5.data
from datasets import load_dataset
from t5.data import postprocessors
from t5.data import preprocessors
from t5.evaluation import metrics
from seqio import FunctionDataSource, utils

TaskRegistry = seqio.TaskRegistry

DATASET_NAME = 'NbAiLab/NCC'
DEFAULT_OUTPUT_FEATURES = {
    "inputs": seqio.Feature(
        vocabulary=t5.data.get_default_vocabulary(), add_eos=True,
        required=False),
    "targets": seqio.Feature(
        vocabulary=t5.data.get_default_vocabulary(), add_eos=True)
}


def gen_dataset(split, column="text"):
    dataset = load_dataset(DATASET_NAME)
    while True:
        for item in dataset[str(split)]:
            yield item[column]


def dataset_fn(split, shuffle_files, seed=None):
    return tf.data.Dataset.from_generator(functools.partial(gen_dataset, split), output_signature=tf.TensorSpec(shape=(), dtype=tf.string, name=DATASET_NAME))


@utils.map_over_dataset
def target_to_key(x, key_map, target_key):
    """Assign the value from the dataset to target_key in key_map"""
    return {**key_map, target_key: x}


# ==================================== C4 ======================================
# Final pretraining task used in Raffel et al., 2019 adaptated to NCC
TaskRegistry.add(
    "c4_v220_span_corruption_ncc",
    source=seqio.FunctionDataSource(dataset_fn=dataset_fn, splits=("train", "validation"), caching_permitted=False),
    preprocessors=[
        functools.partial(
            target_to_key, key_map={
                "inputs": None,
                "targets": None,
            }, target_key="targets"),
        seqio.preprocessors.tokenize,
        #seqio.CacheDatasetPlaceholder(),
        preprocessors.span_corruption,
        seqio.preprocessors.append_eos_after_trim,
    ],
    output_features={"targets": DEFAULT_OUTPUT_FEATURES["targets"]},
    metric_fns=[]
)
