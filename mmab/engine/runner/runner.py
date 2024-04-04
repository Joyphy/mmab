# Copyright (c) OpenMMLab. All rights reserved.
import copy
import logging
import os
import os.path as osp
import pickle
import platform
import time
import warnings
from collections import OrderedDict
from functools import partial
from typing import Callable, Dict, List, Optional, Sequence, Union

import torch
import torch.nn as nn
from torch.nn.parallel.distributed import DistributedDataParallel
from torch.optim import Optimizer
from torch.utils.data import DataLoader

import mmengine
from mmengine.config import Config, ConfigDict
from mmengine.dataset import worker_init_fn as default_worker_init_fn
from mmengine.device import get_device
from mmengine.dist import (broadcast, get_dist_info, get_rank, init_dist,
                           is_distributed, master_only)
from mmengine.evaluator import Evaluator
from mmengine.fileio import FileClient, join_path
from mmengine.hooks import Hook
from mmengine.logging import MessageHub, MMLogger, print_log
from mmengine.model import (MMDistributedDataParallel, convert_sync_batchnorm,
                            is_model_wrapper, revert_sync_batchnorm)
from mmengine.model.efficient_conv_bn_eval import \
    turn_on_efficient_conv_bn_eval
from mmengine.optim import (OptimWrapper, OptimWrapperDict, _ParamScheduler,
                            build_optim_wrapper)
from mmengine.registry import (DATA_SAMPLERS, DATASETS, EVALUATOR, FUNCTIONS,
                               HOOKS, LOG_PROCESSORS, LOOPS, MODEL_WRAPPERS,
                               MODELS, OPTIM_WRAPPERS, PARAM_SCHEDULERS,
                               RUNNERS, VISUALIZERS, DefaultScope)
from mmengine.utils import apply_to, digit_version, get_git_hash, is_seq_of
from mmengine.utils.dl_utils import (TORCH_VERSION, collect_env,
                                     set_multi_processing)
from mmengine.visualization import Visualizer

from mmengine.runner import Runner
from mmab.registry import RUNNERS

ConfigType = Union[Dict, Config, ConfigDict]

@RUNNERS.register_module()
class MemoryBankRunner(Runner):
    def __init__(
        self,
        model: Union[nn.Module, Dict],
        work_dir: str,
        train_dataloader: Optional[Union[DataLoader, Dict]] = None,
        val_dataloader: Optional[Union[DataLoader, Dict]] = None,
        test_dataloader: Optional[Union[DataLoader, Dict]] = None,
        train_cfg: Optional[Dict] = None,
        val_cfg: Optional[Dict] = None,
        test_cfg: Optional[Dict] = None,
        auto_scale_lr: Optional[Dict] = None,
        val_evaluator: Optional[Union[Evaluator, Dict, List]] = None,
        test_evaluator: Optional[Union[Evaluator, Dict, List]] = None,
        default_hooks: Optional[Dict[str, Union[Hook, Dict]]] = None,
        custom_hooks: Optional[List[Union[Hook, Dict]]] = None,
        data_preprocessor: Union[nn.Module, Dict, None] = None,
        load_from: Optional[str] = None,
        resume: bool = False,
        launcher: str = 'none',
        env_cfg: Dict = dict(dist_cfg=dict(backend='nccl')),
        log_processor: Optional[Dict] = None,
        log_level: str = 'INFO',
        visualizer: Optional[Union[Visualizer, Dict]] = None,
        default_scope: str = 'mmengine',
        randomness: Dict = dict(seed=None),
        experiment_name: Optional[str] = None,
        cfg: Optional[ConfigType] = None,
    ):
        self._work_dir = osp.abspath(work_dir)
        mmengine.mkdir_or_exist(self._work_dir)

        # recursively copy the `cfg` because `self.cfg` will be modified
        # everywhere.
        if cfg is not None:
            if isinstance(cfg, Config):
                self.cfg = copy.deepcopy(cfg)
            elif isinstance(cfg, dict):
                self.cfg = Config(cfg)
        else:
            self.cfg = Config(dict())

        # lazy initialization
        training_related = [train_dataloader, train_cfg]
        if not (all(item is None for item in training_related)
                or all(item is not None for item in training_related)):
            raise ValueError(
                'train_dataloader, train_cfg should be '
                'either all None or not None, but got '
                f'train_dataloader={train_dataloader}, '
                f'train_cfg={train_cfg}.')
        self._train_dataloader = train_dataloader
        self._train_loop = train_cfg

        self.auto_scale_lr = auto_scale_lr

        val_related = [val_dataloader, val_cfg, val_evaluator]
        if not (all(item is None
                    for item in val_related) or all(item is not None
                                                    for item in val_related)):
            raise ValueError(
                'val_dataloader, val_cfg, and val_evaluator should be either '
                'all None or not None, but got '
                f'val_dataloader={val_dataloader}, val_cfg={val_cfg}, '
                f'val_evaluator={val_evaluator}')
        self._val_dataloader = val_dataloader
        self._val_loop = val_cfg
        self._val_evaluator = val_evaluator

        test_related = [test_dataloader, test_cfg, test_evaluator]
        if not (all(item is None for item in test_related)
                or all(item is not None for item in test_related)):
            raise ValueError(
                'test_dataloader, test_cfg, and test_evaluator should be '
                'either all None or not None, but got '
                f'test_dataloader={test_dataloader}, test_cfg={test_cfg}, '
                f'test_evaluator={test_evaluator}')
        self._test_dataloader = test_dataloader
        self._test_loop = test_cfg
        self._test_evaluator = test_evaluator

        self._launcher = launcher
        if self._launcher == 'none':
            self._distributed = False
        else:
            self._distributed = True

        # self._timestamp will be set in the `setup_env` method. Besides,
        # it also will initialize multi-process and (or) distributed
        # environment.
        self.setup_env(env_cfg)
        # self._deterministic and self._seed will be set in the
        # `set_randomness`` method
        self._randomness_cfg = randomness
        self.set_randomness(**randomness)

        if experiment_name is not None:
            self._experiment_name = f'{experiment_name}_{self._timestamp}'
        elif self.cfg.filename is not None:
            filename_no_ext = osp.splitext(osp.basename(self.cfg.filename))[0]
            self._experiment_name = f'{filename_no_ext}_{self._timestamp}'
        else:
            self._experiment_name = self.timestamp
        self._log_dir = osp.join(self.work_dir, self.timestamp)
        mmengine.mkdir_or_exist(self._log_dir)
        # Used to reset registries location. See :meth:`Registry.build` for
        # more details.
        if default_scope is not None:
            default_scope = DefaultScope.get_instance(  # type: ignore
                self._experiment_name,
                scope_name=default_scope)
        self.default_scope = default_scope

        # Build log processor to format message.
        log_processor = dict() if log_processor is None else log_processor
        self.log_processor = self.build_log_processor(log_processor)
        # Since `get_instance` could return any subclass of ManagerMixin. The
        # corresponding attribute needs a type hint.
        self.logger = self.build_logger(log_level=log_level)

        # Collect and log environment information.
        self._log_env(env_cfg)

        # Build `message_hub` for communication among components.
        # `message_hub` can store log scalars (loss, learning rate) and
        # runtime information (iter and epoch). Those components that do not
        # have access to the runner can get iteration or epoch information
        # from `message_hub`. For example, models can get the latest created
        # `message_hub` by
        # `self.message_hub=MessageHub.get_current_instance()` and then get
        # current epoch by `cur_epoch = self.message_hub.get_info('epoch')`.
        # See `MessageHub` and `ManagerMixin` for more details.
        self.message_hub = self.build_message_hub()
        # visualizer used for writing log or visualizing all kinds of data
        self.visualizer = self.build_visualizer(visualizer)
        if self.cfg:
            self.visualizer.add_config(self.cfg)

        self._load_from = load_from
        self._resume = resume
        # flag to mark whether checkpoint has been loaded or resumed
        self._has_loaded = False

        # build a model
        if isinstance(model, dict) and data_preprocessor is not None:
            # Merge the data_preprocessor to model config.
            model.setdefault('data_preprocessor', data_preprocessor)
        self.model = self.build_model(model)
        # wrap model
        self.model = self.wrap_model(
            self.cfg.get('model_wrapper_cfg'), self.model)

        # get model name from the model class
        if hasattr(self.model, 'module'):
            self._model_name = self.model.module.__class__.__name__
        else:
            self._model_name = self.model.__class__.__name__

        self._hooks: List[Hook] = []
        # register hooks to `self._hooks`
        self.register_hooks(default_hooks, custom_hooks)
        # log hooks information
        self.logger.info(f'Hooks will be executed in the following '
                         f'order:\n{self.get_hooks_info()}')

        # dump `cfg` to `work_dir`
        self.dump_config()

    @classmethod
    def from_cfg(cls, cfg: ConfigType) -> 'Runner':
        """Build a runner from config.

        Args:
            cfg (ConfigType): A config used for building runner. Keys of
                ``cfg`` can see :meth:`__init__`.

        Returns:
            Runner: A runner build from ``cfg``.
        """
        cfg = copy.deepcopy(cfg)
        runner = cls(
            model=cfg['model'],
            work_dir=cfg['work_dir'],
            train_dataloader=cfg.get('train_dataloader'),
            val_dataloader=cfg.get('val_dataloader'),
            test_dataloader=cfg.get('test_dataloader'),
            train_cfg=cfg.get('train_cfg'),
            val_cfg=cfg.get('val_cfg'),
            test_cfg=cfg.get('test_cfg'),
            auto_scale_lr=cfg.get('auto_scale_lr'),
            val_evaluator=cfg.get('val_evaluator'),
            test_evaluator=cfg.get('test_evaluator'),
            default_hooks=cfg.get('default_hooks'),
            custom_hooks=cfg.get('custom_hooks'),
            data_preprocessor=cfg.get('data_preprocessor'),
            load_from=cfg.get('load_from'),
            resume=cfg.get('resume', False),
            launcher=cfg.get('launcher', 'none'),
            env_cfg=cfg.get('env_cfg'),  # type: ignore
            log_processor=cfg.get('log_processor'),
            log_level=cfg.get('log_level', 'INFO'),
            visualizer=cfg.get('visualizer'),
            default_scope=cfg.get('default_scope', 'mmengine'),
            randomness=cfg.get('randomness', dict(seed=None)),
            experiment_name=cfg.get('experiment_name'),
            cfg=cfg,
        )

        return runner

    def train(self):
        """Launch training.

        Returns:
            nn.Module: The model after training.
        """
        if is_model_wrapper(self.model):
            ori_model = self.model.module
        else:
            ori_model = self.model
        assert hasattr(ori_model, 'train_step'), (
            'If you want to train your model, please make sure your model '
            'has implemented `train_step`.')

        if self._val_loop is not None:
            assert hasattr(ori_model, 'val_step'), (
                'If you want to validate your model, please make sure your '
                'model has implemented `val_step`.')

        if self._train_loop is None:
            raise RuntimeError(
                '`self._train_loop` should not be None when calling train '
                'method. Please provide `train_dataloader`, `train_cfg`, '
                '`optimizer` and `param_scheduler` arguments when '
                'initializing runner.')

        self._train_loop = self.build_train_loop(
            self._train_loop)  # type: ignore

        if self._val_loop is not None:
            self._val_loop = self.build_val_loop(
                self._val_loop)  # type: ignore
        # TODO: add a contextmanager to avoid calling `before_run` many times
        self.call_hook('before_run')

        # initialize the model weights
        self._init_model_weights()

        # try to enable efficient_conv_bn_eval feature
        modules = self.cfg.get('efficient_conv_bn_eval', None)
        if modules is not None:
            self.logger.info(f'Enabling the "efficient_conv_bn_eval" feature'
                             f' for sub-modules: {modules}')
            turn_on_efficient_conv_bn_eval(ori_model, modules)

        # make sure checkpoint-related hooks are triggered after `before_run`
        self.load_or_resume()

        # Maybe compile the model according to options in self.cfg.compile
        # This must be called **AFTER** model has been wrapped.
        self._maybe_compile('train_step')

        model = self.train_loop.run()  # type: ignore
        self.call_hook('after_run')
        return model