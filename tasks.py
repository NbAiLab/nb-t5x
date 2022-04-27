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

DEFAULT_OUTPUT_FEATURES = {
    "inputs": seqio.Feature(
        vocabulary=t5.data.get_default_vocabulary(), add_eos=True,
        required=False),
    "targets": seqio.Feature(
        vocabulary=t5.data.get_default_vocabulary(), add_eos=True)
}
# Custom vocabs can also be defined and loaded
# vocabulary = seqio.SentencePieceVocabulary("gs://t5-data/vocabs/mc4.250000.100extra/sentencepiece.model")


def gen_dataset(split, shuffle=False, seed=None, column="text", dataset_params=None):
    dataset = load_dataset(**dataset_params)
    if shuffle:
        if seed:
            dataset = dataset.shuffle(seed=seed)
        else:
            dataset = dataset.shuffle()
    while True:
        for item in dataset[str(split)]:
            yield item[column]


def dataset_fn(split, shuffle_files, seed=None, dataset_params=None):
    return tf.data.Dataset.from_generator(
        functools.partial(gen_dataset, split, shuffle_files, seed, dataset_params=dataset_params),
        output_signature=tf.TensorSpec(shape=(), dtype=tf.string, name=dataset_name)
    )


@utils.map_over_dataset
def target_to_key(x, key_map, target_key):
    """Assign the value from the dataset to target_key in key_map"""
    return {**key_map, target_key: x}


# Final pretraining task used in Raffel et al., 2019 adaptated to NCC
dataset_name = 'NbAiLab/NCC'
dataset_params = {"path": dataset_name}
dataset_shapes = {'train': 20830348, 'validation': 473079}
TaskRegistry.add(
    f"c4_v220_span_corruption_{dataset_name.lower().replace('/', '_').replace('.', '')}",
    source=seqio.FunctionDataSource(
        dataset_fn=functools.partial(dataset_fn, dataset_params=dataset_params),
        splits=("train", "validation"),
        caching_permitted=False,
        num_input_examples=dataset_shapes,
    ),
    preprocessors=[
        functools.partial(
            target_to_key, key_map={
                "inputs": None,
                "targets": None,
            }, target_key="targets"),
        seqio.preprocessors.tokenize,
        # seqio.CacheDatasetPlaceholder(),
        preprocessors.span_corruption,
        seqio.preprocessors.append_eos_after_trim,
    ],
    output_features={"targets": DEFAULT_OUTPUT_FEATURES["targets"]},
    metric_fns=[]
)

# Final pretraining task used in Raffel et al., 2019 adaptated to nbailab_extended
dataset_name = 'NbAiLab/nbailab_extended'
dataset_params = {"path": dataset_name, "use_auth_token": True, "streaming": True}
dataset_shapes = None
TaskRegistry.add(
    f"c4_v220_span_corruption_{dataset_name.lower().replace('/', '_').replace('.', '')}",
    source=seqio.FunctionDataSource(
        dataset_fn=functools.partial(dataset_fn, dataset_params=dataset_params),
        splits=("train", "validation"),
        caching_permitted=False,
        num_input_examples=dataset_shapes,
    ),
    preprocessors=[
        functools.partial(
            target_to_key, key_map={
                "inputs": None,
                "targets": None,
            }, target_key="targets"),
        seqio.preprocessors.tokenize,
        # seqio.CacheDatasetPlaceholder(),
        preprocessors.span_corruption,
        seqio.preprocessors.append_eos_after_trim,
    ],
    output_features={"targets": DEFAULT_OUTPUT_FEATURES["targets"]},
    metric_fns=[]
)

# Final pretraining task used in Raffel et al., 2019 adaptated to NCC
dataset_name = 'NbAiLab/NCC'
dataset_params = {"path": dataset_name}
dataset_shapes = {'train': 20830348, 'validation': 473079}
vocabulary = seqio.SentencePieceVocabulary("gs://t5-data/vocabs/mc4.250000.100extra/sentencepiece.model")
TaskRegistry.add(
    "ncc_span_corruption_mt5",
    source=seqio.FunctionDataSource(
        dataset_fn=functools.partial(dataset_fn, dataset_params=dataset_params),
        splits=("train", "validation"),
        caching_permitted=False,
        num_input_examples=dataset_shapes,
    ),
    preprocessors=[
        functools.partial(
            target_to_key, key_map={
                "inputs": None,
                "targets": None,
            }, target_key="targets"),
        seqio.preprocessors.tokenize,
        # seqio.CacheDatasetPlaceholder(),
        preprocessors.span_corruption, 
        seqio.preprocessors.append_eos_after_trim,
    ],
    output_features={"targets": seqio.Feature(vocabulary=vocabulary, add_eos=True)},
    metric_fns=[]
)

# Final pretraining task used in Raffel et al., 2019 adaptated to NCC
dataset_name = 'NbAiLab/nbailab_extended'
dataset_params = {"path": dataset_name, "use_auth_token": True, "streaming": True}
dataset_shapes = None
vocabulary = seqio.SentencePieceVocabulary("gs://t5-data/vocabs/mc4.250000.100extra/sentencepiece.model")
TaskRegistry.add(
    "extended_span_corruption_mt5",
    source=seqio.FunctionDataSource(
        dataset_fn=functools.partial(dataset_fn, dataset_params=dataset_params),
        splits=("train", "validation"),
        caching_permitted=False,
        num_input_examples=dataset_shapes,
    ),
    preprocessors=[
        functools.partial(
            target_to_key, key_map={
                "inputs": None,
                "targets": None,
            }, target_key="targets"),
        seqio.preprocessors.tokenize,
        # seqio.CacheDatasetPlaceholder(),
        preprocessors.span_corruption,
        seqio.preprocessors.append_eos_after_trim,
    ],
    output_features={"targets": seqio.Feature(vocabulary=vocabulary, add_eos=True)},
    metric_fns=[]
)

# Final pretraining task used in Raffel et al., 2019 adaptated to NCC
dataset_name = 'NbAiLab/nbailab_extended'
dataset_params = {"path": dataset_name, "use_auth_token": True, "streaming": True}
dataset_shapes = None
vocabulary = seqio.SentencePieceVocabulary("gs://nb-t5/t5/vocabs/oscar/no_32000_bpe.sp.model", extra_ids=100)
TaskRegistry.add(
    "extended_span_corruption_pretrain",
    source=seqio.FunctionDataSource(
        dataset_fn=functools.partial(dataset_fn, dataset_params=dataset_params),
        splits=("train", "validation"),
        caching_permitted=False,
        num_input_examples=dataset_shapes,
    ),
    preprocessors=[
        functools.partial(
            target_to_key, key_map={
                "inputs": None,
                "targets": None,
            }, target_key="targets"),
        seqio.preprocessors.tokenize,
        # seqio.CacheDatasetPlaceholder(),
        preprocessors.span_corruption,
        seqio.preprocessors.append_eos_after_trim,
    ],
    output_features={"targets": seqio.Feature(vocabulary=vocabulary, add_eos=True)},
    metric_fns=[]
)
