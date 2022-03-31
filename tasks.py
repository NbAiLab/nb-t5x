"""Add Tasks to registry."""
import functools

import seqio
import t5.data
from datasets import load_dataset
from t5.data import postprocessors
from t5.data import preprocessors
from t5.evaluation import metrics
#import tensorflow as tf
import tensorflow.compat.v2 as tf
import tensorflow_datasets as tfds

TaskRegistry = seqio.TaskRegistry

DEFAULT_OUTPUT_FEATURES = {
#    "inputs": seqio.Feature(
#        vocabulary=t5.data.get_default_vocabulary(), add_eos=True,
#        required=False),
    "targets": seqio.Feature(
        vocabulary=t5.data.get_default_vocabulary(), add_eos=True)
}



from seqio import FunctionDataSource, utils

# ncc = load_dataset('NbAiLab/NCC')

def gen_ncc(split):
    ncc = load_dataset('NbAiLab/NCC')
    while True:
        for item in ncc[str(split)]:
            #print(item["text"])
            yield item["text"]


def dataset_fn(split, shuffle_files, seed=None):
    #shape = {'train': 20830348, 'validation': 473079}
    return tf.data.Dataset.from_generator(functools.partial(gen_ncc, split), output_signature=tf.TensorSpec(shape=(), dtype=tf.string, name="NCC"))


@utils.map_over_dataset
def value_to_dict(x, key_map, value_key):
    """Assign the value from the dataset to value_key in key_map"""
    # print(type(x), x, str(x))
    key_map[value_key] = str(x)  # .numpy().decode("utf8")
    return key_map


# ==================================== C4 ======================================
# Final pretraining task used in Raffel et al., 2019 adaptated to NCC
TaskRegistry.add(
    "c4_v220_span_corruption_ncc",
    source=seqio.FunctionDataSource(dataset_fn=dataset_fn, splits=("train", "validation"), caching_permitted=False),
    preprocessors=[
        functools.partial(
            value_to_dict, key_map={
                "inputs": None,
                "targets": None,
            }, value_key="targets"),
        seqio.preprocessors.tokenize,
        #seqio.CacheDatasetPlaceholder(),
        preprocessors.span_corruption,
        seqio.preprocessors.append_eos_after_trim,

    ],
    output_features=DEFAULT_OUTPUT_FEATURES,
#    output_features=seqio.Feature(
#        vocabulary=t5.data.get_default_vocabulary(), add_eos=True),
    metric_fns=[]
)
