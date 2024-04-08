from mmengine.registry import RUNNERS as MMENGINE_RUNNERS
from mmengine.registry import LOOPS as MMENGINE_LOOPS
from mmengine.registry import DATASETS as MMENGINE_DATASETS
from mmengine.registry import TRANSFORMS as MMENGINE_TRANSFORMS
from mmengine.registry import MODELS as MMENGINE_MODELS
from mmengine.registry import METRICS as MMENGINE_METRICS 
from mmengine.registry import HOOKS as MMENGINE_HOOKS
from mmengine.registry import Registry

RUNNERS = Registry(
    "runners", parent=MMENGINE_RUNNERS, locations=["mmab.engine.runner"])

LOOPS = Registry(
    "loops", parent=MMENGINE_LOOPS, locations=["mmab.engine.runner"])

DATASETS = Registry(
    "datasets", parent=MMENGINE_DATASETS, locations=["mmab.datasets"])

TRANSFORMS = Registry(
    "transforms", parent=MMENGINE_TRANSFORMS, locations=["mmab.datasets.transforms"])

MODELS = Registry(
    "models", parent=MMENGINE_MODELS, locations=["mmab.models"])

METRICS = Registry(
    "metrics", parent=MMENGINE_METRICS, locations=["mmab.evaluation"])

HOOKS = Registry(
    "hooks", parent=MMENGINE_HOOKS, locations=["mmab.engine.hooks"])