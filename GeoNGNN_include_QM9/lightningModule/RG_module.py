from typing import Any
import pytorch_lightning as pl
from torch import Tensor
from utils.loss_fns import loss_fn_map
from utils.activation_fns import activation_fn_map
from utils.EMA import ExponentialMovingAverage
import torch
from torch.nn.utils.clip_grad import clip_grad_norm_
from lightningModule.basic_procedures import configure_optimizers_




class RG_module(pl.LightningModule):
    def __init__(
        self, 
        model_config: dict,
        optimizer_config: dict,
        scheduler_config: dict,
        global_y_std: float,
        global_y_mean: float,
        C: float,
        subg_C: float,
        model,
        test_dataset_name=None,
        output_layer_type="linear",
        **kwargs
        ):
        
        super().__init__()
        self.save_hyperparameters()
        self.model = model(
                z_hidden_dim=model_config.z_hidden_dim,
                hidden_dim=model_config.hidden_dim,
                ef_dim=model_config.ef_dim,
                rbf=model_config.rbf,
                max_z=model_config.max_z,
                outer_rbound_upper=model_config.cutoff + 2.0,
                inner_rbound_upper=model_config.subg_cutoff + 2.0,
                activation_fn=activation_fn_map[model_config.activation_fn],
                inner_layer_num=model_config.inner_layer_num,
                outer_layer_num=model_config.outer_layer_num,
                inner_cutoff=model_config.subg_cutoff,
                outer_cutoff=model_config.cutoff,
                predict_force=False,
                global_y_mean=global_y_mean,
                global_y_std=global_y_std,
                C=C,
                subg_C=subg_C,
                ablation_innerGNN=model_config.ablation_innerGNN,
                extend_r=model_config.extend_r,
                output_layer_type=output_layer_type
            )
        
        # ema configs
        self.ema_model = ExponentialMovingAverage(self, decay=optimizer_config.ema_decay, device=self.device)
        
        self.test_dataset_name = test_dataset_name
        # optimizer and scheduler configs
        self.automatic_optimization = False
        
        self.optimizer_config = optimizer_config
        self.scheduler_config = scheduler_config
        
        self.force_ratio = optimizer_config.force_ratio
        self.gradient_clip_val = optimizer_config.gradient_clip_val
        self.warmup_end = False
        self.warmup_epoch = scheduler_config.warmup_epoch
        
        self.train_loss_fn = loss_fn_map["l2"]
        self.metric_fn = loss_fn_map["l1"]

        
        
        
    def forward(self, batch_data):
        return self.model(batch_data)
        
    def configure_optimizers(self):
        return configure_optimizers_(self)
            
    def training_step(self, batch, batch_idx):

        loss = self.reg_step_once(batch, train=True)
        batch_size = batch.z.shape[0]

        # log multiple loss
        self.log('train_loss/train_loss', loss, prog_bar=True, on_epoch=True, on_step=True, sync_dist=True, batch_size=batch_size)
        

        # optimize manually
        if loss != loss:
            print("nan loss, skip this step")
            return loss
        self.opt_step(loss)
        
        # log
        self.log("learning_rate/lr_rate_AdamW", self.optimizers().param_groups[0]["lr"], on_epoch=True, on_step=False, sync_dist=True, batch_size=batch_size)
        
        return loss
        
        
    def validation_step(self, batch, batch_idx):

        
        torch.set_grad_enabled(True)
        
        loss = self.reg_step_once(batch, train=False)
        batch_size = batch.z.shape[0]

        # log multiple loss
        self.log('val_loss/val_loss', loss, prog_bar=True, on_epoch=True, on_step=False, sync_dist=True, batch_size=batch_size)
        
        return loss
        
        
    def test_step(self, batch, batch_idx):



        with torch.inference_mode(False):
            import easydict
            cloned_batch = easydict.EasyDict()
            for name in batch.keys():
                try:
                    cloned_batch[name] = getattr(batch, name).clone()
                except:
                    cloned_batch[name] = getattr(batch, name)
            batch = cloned_batch
            batch_size = batch.z.shape[0]

            loss = self.reg_step_once(batch, train=False)
            
                
            self.log("test_loss/test_loss", loss, sync_dist=True, batch_size=batch_size)
                
            return loss

    def on_train_batch_end(self, outputs, batch, batch_idx):
        self.ema_model.update_parameters(self) 
        

    def on_train_epoch_end(self):
        
        if self.warmup_end == False:
            if self.current_epoch >= self.warmup_epoch:
                self.warmup_end = True
            
        for scheduler in self.lr_schedulers():
            if not isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step()
                
                # update lr, which is necessary since torch-lightning does not automatically update the lr when restoring the model
                lr = scheduler.get_last_lr()[0]
                self.optimizers().optimizer.param_groups[0]["lr"] = lr
                self.optimizers().param_groups[0]["lr"] = lr


    def on_validation_epoch_end(self):
        if self.warmup_end:
            for scheduler in self.lr_schedulers():
                if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    scheduler.step(self.trainer.callback_metrics[self.optimizer_config.monitor])
                    
                    # update lr, which is necessary since torch-lightning does not automatically update the lr when restoring the model
                    lr = scheduler.optimizer.param_groups[0]["lr"]
                    self.optimizers().optimizer.param_groups[0]["lr"] = lr
                    self.optimizers().param_groups[0]["lr"] = lr
                    self.lr_schedulers()[1].after_scheduler.optimizer.param_groups[0]["lr"] = lr

    def opt_step(self, loss) -> None:
        self.optimizers().zero_grad()
        self.manual_backward(loss)
        clip_grad_norm_(self.parameters(), self.gradient_clip_val)
        self.optimizers().step()
        


    def reg_step_once(self, batch, train=False):
        
        if train:
            model = self
            loss_fn = self.train_loss_fn
        else:
            model = self.ema_model
            loss_fn = self.metric_fn
        
        pred_y = model(batch)
        pred_y = pred_y.squeeze()
        
        y = batch.y.squeeze()
                
        loss = loss_fn(pred_y, y)
        
        if train:
            return loss
        else:
            return loss.detach()
        
        