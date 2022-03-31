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
DATASET_PARAMS = {}
DATASET_SHAPES = {'train': 20830348, 'validation': 473079}

DEFAULT_OUTPUT_FEATURES = {
    "inputs": seqio.Feature(
        vocabulary=t5.data.get_default_vocabulary(), add_eos=True,
        required=False),
    "targets": seqio.Feature(
        vocabulary=t5.data.get_default_vocabulary(), add_eos=True)
}


def gen_dataset(split, shuffle=False, seed=None, column="text"):
    dataset = load_dataset(DATASET_NAME, **DATASET_PARAMS)
    if shuffle:
        if seed:
            dataset = dataset.shuffle(seed=seed)
        else:
            dataset = dataset.shuffle()
    while True:
        for item in dataset[str(split)]:
            yield item[column]


def dataset_fn(split, shuffle_files, seed=None):
    return tf.data.Dataset.from_generator(
        functools.partial(gen_dataset, split, shuffle_files, seed),
        output_signature=tf.TensorSpec(shape=(), dtype=tf.string, name=DATASET_NAME)
    )


@utils.map_over_dataset
def target_to_key(x, key_map, target_key):
    """Assign the value from the dataset to target_key in key_map"""
    return {**key_map, target_key: x}


# ==================================== C4 ======================================
# Final pretraining task used in Raffel et al., 2019 adaptated to NCC
TaskRegistry.add(
    f"c4_v220_span_corruption_{DATASET_NAME.lower().replace('/', '_').replace('.', '')}",
    source=seqio.FunctionDataSource(dataset_fn=dataset_fn, splits=("train", "validation"), caching_permitted=True, num_input_examples=DATASET_SHAPES),
    preprocessors=[
        functools.partial(
            target_to_key, key_map={
                "inputs": None,
                "targets": None,
            }, target_key="targets"),
        seqio.preprocessors.tokenize,
        seqio.CacheDatasetPlaceholder(),
        preprocessors.span_corruption,
        seqio.preprocessors.append_eos_after_trim,
    ],
    output_features={"targets": DEFAULT_OUTPUT_FEATURES["targets"]},
    metric_fns=[]
)
