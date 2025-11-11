if __name__ == "__main__":
    import sys
    import os
    import pathlib

    ROOT_DIR = str(pathlib.Path(__file__).parent.parent.parent)
    sys.path.append(ROOT_DIR)
    os.chdir(ROOT_DIR)

import os
import hydra
import torch
from omegaconf import OmegaConf
import pathlib
from torch.utils.data import DataLoader
import copy
import random
import wandb
import pickle
import tqdm
import numpy as np
import shutil
from diffusion_policy.workspace.base_workspace import BaseWorkspace
from diffusion_policy.policy.diffusion_unet_image_policy import DiffusionUnetImagePolicy
from diffusion_policy.dataset.base_dataset import BaseImageDataset, BaseDataset
from diffusion_policy.env_runner.base_image_runner import BaseImageRunner
from diffusion_policy.common.checkpoint_util import TopKCheckpointManager
from diffusion_policy.common.json_logger import JsonLogger
from diffusion_policy.common.pytorch_util import dict_apply, optimizer_to
from diffusion_policy.model.diffusion.ema_model import EMAModel
from diffusion_policy.model.common.lr_scheduler import get_scheduler
from accelerate import Accelerator

OmegaConf.register_new_resolver("eval", eval, replace=True)

class TrainDiffusionUnetFADPWorkspace(BaseWorkspace):
    """
    Workspace for training FADP (Fuxin Action Diffusion Policy) models.
    
    This workspace is adapted from TrainDiffusionUnetImageWorkspace but removes
    the dependency on the 'pretrained' field in obs_encoder config, making it
    compatible with custom encoders like FADPEncoder that don't use this field.
    """
    include_keys = ['global_step', 'epoch']
    exclude_keys = tuple()

    def __init__(self, cfg: OmegaConf, output_dir=None):
        super().__init__(cfg, output_dir=output_dir)

        # set seed
        seed = cfg.training.seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        # configure model
        self.model: DiffusionUnetImagePolicy = hydra.utils.instantiate(cfg.policy)

        self.ema_model: DiffusionUnetImagePolicy = None
        if cfg.training.use_ema:
            self.ema_model = copy.deepcopy(self.model)

        # configure optimizer with separate learning rates for encoder and model
        obs_encoder_lr = cfg.optimizer.lr
        
        # Check if dinov3_frozen field exists and adjust lr accordingly
        # For frozen encoders, we reduce the learning rate
        if hasattr(cfg.policy.obs_encoder, 'dinov3_frozen'):
            if cfg.policy.obs_encoder.dinov3_frozen:
                obs_encoder_lr *= 0.1
                print('==> DINOv3 encoder is frozen, reducing lr to', obs_encoder_lr)
            else:
                print('==> DINOv3 encoder is trainable')
        
        # Collect encoder parameters
        obs_encoder_params = list()
        for param in self.model.obs_encoder.parameters():
            if param.requires_grad:
                obs_encoder_params.append(param)
        print(f'obs_encoder params: {len(obs_encoder_params)}')
        
        # Setup parameter groups with different learning rates
        param_groups = [
            {'params': self.model.model.parameters()},
            {'params': obs_encoder_params, 'lr': obs_encoder_lr}
        ]
        
        optimizer_cfg = OmegaConf.to_container(cfg.optimizer, resolve=True)
        optimizer_cfg.pop('_target_')
        self.optimizer = torch.optim.AdamW(
            params=param_groups,
            **optimizer_cfg
        )

        # configure training state
        self.global_step = 0
        self.epoch = 0

        # do not save optimizer if resume=False
        if not cfg.training.resume:
            self.exclude_keys = ['optimizer']

    def run(self):
        cfg = copy.deepcopy(self.cfg)

        accelerator = Accelerator(log_with='wandb')
        wandb_cfg = OmegaConf.to_container(cfg.logging, resolve=True)
        wandb_cfg.pop('project')
        accelerator.init_trackers(
            project_name=cfg.logging.project,
            config=OmegaConf.to_container(cfg, resolve=True),
            init_kwargs={"wandb": wandb_cfg}
        )

        # resume training
        if cfg.training.resume:
            lastest_ckpt_path = self.get_checkpoint_path()
            if lastest_ckpt_path.is_file():
                accelerator.print(f"Resuming from checkpoint {lastest_ckpt_path}")
                self.load_checkpoint(path=lastest_ckpt_path)

        # configure dataset
        dataset: BaseImageDataset
        dataset = hydra.utils.instantiate(cfg.task.dataset)
        assert isinstance(dataset, BaseImageDataset) or isinstance(dataset, BaseDataset)
        train_dataloader = DataLoader(dataset, **cfg.dataloader)

        # compute normalizer on the main process and save to disk
        normalizer_path = os.path.join(self.output_dir, 'normalizer.pkl')
        if accelerator.is_main_process:
            normalizer = dataset.get_normalizer()
            pickle.dump(normalizer, open(normalizer_path, 'wb'))

        # load normalizer on all processes
        accelerator.wait_for_everyone()
        normalizer = pickle.load(open(normalizer_path, 'rb'))

        # configure validation dataset
        val_dataset = dataset.get_validation_dataset()
        val_dataloader = DataLoader(val_dataset, **cfg.val_dataloader)
        print('train dataset:', len(dataset), 'train dataloader:', len(train_dataloader))
        print('val dataset:', len(val_dataset), 'val dataloader:', len(val_dataloader))

        self.model.set_normalizer(normalizer)
        if cfg.training.use_ema:
            self.ema_model.set_normalizer(normalizer)

        # configure lr scheduler
        lr_scheduler = get_scheduler(
            cfg.training.lr_scheduler,
            optimizer=self.optimizer,
            num_warmup_steps=cfg.training.lr_warmup_steps,
            num_training_steps=(
                len(train_dataloader) * cfg.training.num_epochs) \
                    // cfg.training.gradient_accumulate_every,
            # pytorch assumes stepping LRScheduler every epoch
            # however huggingface diffusers steps it every batch
            last_epoch=self.global_step-1
        )

        # configure ema
        ema: EMAModel = None
        if cfg.training.use_ema:
            ema = hydra.utils.instantiate(
                cfg.ema,
                model=self.ema_model)

        # prepare for accelerator
        self.model, self.optimizer, train_dataloader, val_dataloader, lr_scheduler = accelerator.prepare(
            self.model, self.optimizer, train_dataloader, val_dataloader, lr_scheduler
        )

        # device transfer
        device = accelerator.device
        # self.model.to(device)
        if self.ema_model is not None:
            self.ema_model.to(device)

        cfg.logging.name = os.path.basename(self.output_dir)
        # configure checkpoint manager
        topk_manager = TopKCheckpointManager(
            save_dir=os.path.join(self.output_dir, 'checkpoints'),
            **cfg.checkpoint.topk
        )

        # configure logging
        wandb_run = accelerator.get_tracker("wandb")
        wandb_run = wandb_run.run if hasattr(wandb_run, 'run') else wandb_run
        
        # training loop with json logger as context manager
        log_path = os.path.join(self.output_dir, 'logs.json.txt')
        with JsonLogger(log_path) as json_logger:
            self._train_loop(
                cfg=cfg,
                accelerator=accelerator,
                wandb_run=wandb_run,
                json_logger=json_logger,
                topk_manager=topk_manager,
                train_dataloader=train_dataloader,
                val_dataloader=val_dataloader,
                lr_scheduler=lr_scheduler,
                ema=ema,
                device=device
            )
        
        accelerator.end_training()
    
    def _train_loop(
        self,
        cfg,
        accelerator,
        wandb_run,
        json_logger,
        topk_manager,
        train_dataloader,
        val_dataloader,
        lr_scheduler,
        ema,
        device
    ):
        """Training loop implementation"""
        # 保存训练batch用于采样评估
        train_sampling_batch = None
        
        for local_epoch_idx in range(cfg.training.num_epochs):
            step_log = dict()
            # ========= train for this epoch ==========
            if accelerator.is_main_process:
                train_losses = list()
                train_loss_actions = list()
                train_loss_forces = list()
                train_loss_forces_weighted = list()
            with tqdm.tqdm(train_dataloader, desc=f"Training epoch {self.epoch}", 
                          leave=False, mininterval=cfg.training.tqdm_interval_sec) as tepoch:
                for batch_idx, batch in enumerate(tepoch):
                    # 保存最新的batch用于采样
                    train_sampling_batch = batch
                    
                    # device transfer
                    # batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))
                    if batch_idx % cfg.training.gradient_accumulate_every != 0:
                        with accelerator.no_sync(self.model):
                            # forward and compute loss with components
                            loss_dict = self.model.compute_loss(batch, return_components=True)
                            if isinstance(loss_dict, dict):
                                raw_loss = loss_dict['loss']
                            else:
                                raw_loss = loss_dict
                                loss_dict = None
                            loss = raw_loss / cfg.training.gradient_accumulate_every
                            accelerator.backward(loss)
                    else:
                        # forward and compute loss with components
                        loss_dict = self.model.compute_loss(batch, return_components=True)
                        if isinstance(loss_dict, dict):
                            raw_loss = loss_dict['loss']
                        else:
                            raw_loss = loss_dict
                            loss_dict = None
                        loss = raw_loss / cfg.training.gradient_accumulate_every
                        accelerator.backward(loss)
                        self.optimizer.step()
                        self.optimizer.zero_grad()
                        lr_scheduler.step()
                    
                    # update ema
                    if cfg.training.use_ema:
                        ema.step(self.model)

                    # logging
                    raw_loss_cpu = raw_loss.item()
                    tepoch.set_postfix(loss=raw_loss_cpu, refresh=False)
                    if accelerator.is_main_process:
                        train_losses.append(raw_loss_cpu)
                        
                        # 收集loss分量用于epoch平均
                        if loss_dict is not None and isinstance(loss_dict, dict):
                            if 'loss_action' in loss_dict:
                                train_loss_actions.append(loss_dict['loss_action'].item())
                            if 'loss_force' in loss_dict:
                                train_loss_forces.append(loss_dict['loss_force'].item())
                            if 'loss_force_weighted' in loss_dict:
                                train_loss_forces_weighted.append(loss_dict['loss_force_weighted'].item())
                    
                    step_log = {
                        'train_loss': raw_loss_cpu,
                        'global_step': self.global_step,
                        'epoch': self.epoch,
                        'lr': lr_scheduler.get_last_lr()[0]
                    }
                    
                    # 添加loss分量到日志（每个batch都记录）
                    if loss_dict is not None and isinstance(loss_dict, dict):
                        for key in ['loss_action', 'loss_force', 'loss_force_weighted']:
                            if key in loss_dict:
                                step_log[f'train_{key}'] = loss_dict[key].item()

                    is_last_batch = (batch_idx == (len(train_dataloader)-1))
                    if not is_last_batch:
                        # log of last step is combined with validation and rollout
                        if accelerator.is_main_process:
                            wandb_run.log(step_log, step=self.global_step)
                            json_logger.log(step_log)
                            self.global_step += 1

                    if (cfg.training.max_train_steps is not None) \
                        and batch_idx >= (cfg.training.max_train_steps-1):
                        break

            # at the end of each epoch
            # replace train_loss with epoch average
            if accelerator.is_main_process:
                train_loss = np.mean(train_losses)
                step_log['train_loss'] = train_loss
                
                # 添加loss分量的epoch平均值
                if len(train_loss_actions) > 0:
                    step_log['train_loss_action'] = np.mean(train_loss_actions)
                if len(train_loss_forces) > 0:
                    step_log['train_loss_force'] = np.mean(train_loss_forces)
                if len(train_loss_forces_weighted) > 0:
                    step_log['train_loss_force_weighted'] = np.mean(train_loss_forces_weighted)

            # ========= eval for this epoch ==========
            if accelerator.is_main_process:
                policy = self.model
                if cfg.training.use_ema:
                    policy = self.ema_model
                policy.eval()

                # run validation
                if (self.epoch % cfg.training.val_every) == 0:
                    with torch.no_grad():
                        val_losses = list()
                        val_loss_actions = list()
                        val_loss_forces = list()
                        val_loss_forces_weighted = list()
                        with tqdm.tqdm(val_dataloader, desc=f"Validation epoch {self.epoch}", 
                                      leave=False, mininterval=cfg.training.tqdm_interval_sec) as tepoch:
                            for batch_idx, batch in enumerate(tepoch):
                                batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))
                                loss_dict = policy.compute_loss(batch, return_components=True)
                                
                                # 提取loss值
                                if isinstance(loss_dict, dict):
                                    val_losses.append(loss_dict['loss'])
                                    if 'loss_action' in loss_dict:
                                        val_loss_actions.append(loss_dict['loss_action'])
                                    if 'loss_force' in loss_dict:
                                        val_loss_forces.append(loss_dict['loss_force'])
                                    if 'loss_force_weighted' in loss_dict:
                                        val_loss_forces_weighted.append(loss_dict['loss_force_weighted'])
                                else:
                                    val_losses.append(loss_dict)
                                
                                if (cfg.training.max_val_steps is not None) \
                                    and batch_idx >= (cfg.training.max_val_steps-1):
                                    break
                        
                        if len(val_losses) > 0:
                            val_loss = torch.mean(torch.tensor(val_losses)).item()
                            # log epoch average validation loss
                            step_log['val_loss'] = val_loss
                            
                            # log loss components if available
                            if len(val_loss_actions) > 0:
                                step_log['val_loss_action'] = torch.mean(torch.tensor(val_loss_actions)).item()
                            if len(val_loss_forces) > 0:
                                step_log['val_loss_force'] = torch.mean(torch.tensor(val_loss_forces)).item()
                            if len(val_loss_forces_weighted) > 0:
                                step_log['val_loss_force_weighted'] = torch.mean(torch.tensor(val_loss_forces_weighted)).item()
                
                # 定义action MSE计算函数
                def log_action_mse(step_log, category, pred_action, gt_action):
                    """计算并记录action预测误差（支持7维、10维、13维）"""
                    B, T, D = pred_action.shape
                    
                    # 动态检测action维度：7维(axis-angle)、10维(rot6d)、13维(axis-angle+force)
                    if D % 13 == 0:
                        action_dim = 13
                        n_robots = D // 13
                    elif D % 7 == 0:
                        action_dim = 7
                        n_robots = D // 7
                    elif D % 10 == 0:
                        action_dim = 10
                        n_robots = D // 10
                    else:
                        action_dim = D
                        n_robots = 1
                    
                    if action_dim == 13:
                        # 13维格式: 3 pos + 3 rot + 1 gripper + 6 force
                        pred_action = pred_action.view(B, T, n_robots, 13)
                        gt_action = gt_action.view(B, T, n_robots, 13)
                        
                        # 总体MSE
                        step_log[f'{category}_action_mse_error'] = torch.nn.functional.mse_loss(pred_action, gt_action)
                        
                        # Action部分 (前7维)
                        step_log[f'{category}_action_mse_error_pos'] = torch.nn.functional.mse_loss(pred_action[..., :3], gt_action[..., :3])
                        step_log[f'{category}_action_mse_error_rot'] = torch.nn.functional.mse_loss(pred_action[..., 3:6], gt_action[..., 3:6])
                        step_log[f'{category}_action_mse_error_gripper'] = torch.nn.functional.mse_loss(pred_action[..., 6], gt_action[..., 6])
                        
                        # Force部分 (后6维)
                        step_log[f'{category}_force_mse_error'] = torch.nn.functional.mse_loss(pred_action[..., 7:], gt_action[..., 7:])
                        
                    elif action_dim == 7:
                        # 7维格式: 3 pos + 3 rot + 1 gripper
                        pred_action = pred_action.view(B, T, n_robots, 7)
                        gt_action = gt_action.view(B, T, n_robots, 7)
                        step_log[f'{category}_action_mse_error'] = torch.nn.functional.mse_loss(pred_action, gt_action)
                        step_log[f'{category}_action_mse_error_pos'] = torch.nn.functional.mse_loss(pred_action[..., :3], gt_action[..., :3])
                        step_log[f'{category}_action_mse_error_rot'] = torch.nn.functional.mse_loss(pred_action[..., 3:6], gt_action[..., 3:6])
                        step_log[f'{category}_action_mse_error_gripper'] = torch.nn.functional.mse_loss(pred_action[..., 6], gt_action[..., 6])
                        
                    elif action_dim == 10:
                        # 10维格式: 3 pos + 6 rot6d + 1 gripper
                        pred_action = pred_action.view(B, T, n_robots, 10)
                        gt_action = gt_action.view(B, T, n_robots, 10)
                        step_log[f'{category}_action_mse_error'] = torch.nn.functional.mse_loss(pred_action, gt_action)
                        step_log[f'{category}_action_mse_error_pos'] = torch.nn.functional.mse_loss(pred_action[..., :3], gt_action[..., :3])
                        step_log[f'{category}_action_mse_error_rot'] = torch.nn.functional.mse_loss(pred_action[..., 3:9], gt_action[..., 3:9])
                        step_log[f'{category}_action_mse_error_gripper'] = torch.nn.functional.mse_loss(pred_action[..., 9], gt_action[..., 9])
                        
                    else:
                        # 不支持的维度，只记录总MSE
                        step_log[f'{category}_action_mse_error'] = torch.nn.functional.mse_loss(pred_action, gt_action)
                
                # 在训练集样本上运行扩散采样，评估预测误差
                if (self.epoch % cfg.training.get('sample_every', 5)) == 0:
                    with torch.no_grad():
                        # 从训练集采样并评估差异
                        if train_sampling_batch is not None:
                            batch = dict_apply(train_sampling_batch, lambda x: x.to(device, non_blocking=True))
                            gt_action = batch['action']
                            pred_action = policy.predict_action(batch['obs'], None)['action_pred']
                            log_action_mse(step_log, 'train', pred_action, gt_action)
                        
                        # 从验证集采样并评估
                        if len(val_dataloader) > 0:
                            val_sampling_batch = next(iter(val_dataloader))
                            batch = dict_apply(val_sampling_batch, lambda x: x.to(device, non_blocking=True))
                            gt_action = batch['action']
                            pred_action = policy.predict_action(batch['obs'], None)['action_pred']
                            log_action_mse(step_log, 'val', pred_action, gt_action)
                        
                        # 清理内存
                        del batch
                        del gt_action
                        del pred_action

                # checkpoint
                if (self.epoch % cfg.training.checkpoint_every) == 0:
                    # checkpointing
                    if cfg.checkpoint.save_last_ckpt:
                        self.save_checkpoint()
                    if cfg.checkpoint.save_last_snapshot:
                        self.save_snapshot()

                    # sanitize metric names
                    metric_dict = dict()
                    for key, value in step_log.items():
                        new_key = key.replace('/', '_')
                        metric_dict[new_key] = value
                    
                    # We can't copy the last checkpoint here
                    # since save_checkpoint uses threads.
                    # therefore at this point the file might have been empty!
                    topk_ckpt_path = topk_manager.get_ckpt_path(metric_dict)

                    if topk_ckpt_path is not None:
                        self.save_checkpoint(path=topk_ckpt_path)
                # ========= eval end for this epoch ==========
                policy.train()

                # end of epoch
                # log of last step is combined with validation and rollout
                wandb_run.log(step_log, step=self.global_step)
                json_logger.log(step_log)
                self.global_step += 1
                self.epoch += 1

