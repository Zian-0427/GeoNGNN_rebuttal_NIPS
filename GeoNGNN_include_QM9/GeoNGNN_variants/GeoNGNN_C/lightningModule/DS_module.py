import pytorch_lightning as pl
import torch.nn as nn
from utils.loss_fns import loss_fn_map
from utils.activation_fns import activation_fn_map
from utils.EMA import ExponentialMovingAverage
import torch
from torch.nn.utils.clip_grad import clip_grad_norm_
from lightningModule.basic_procedures import configure_optimizers_

class DS_module(pl.LightningModule):
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
        ):
        
        super().__init__()
        self.save_hyperparameters()
        self.classloss = torch.nn.BCELoss()
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
                global_y_mean=global_y_mean,
                global_y_std=global_y_std,
                C=C,
                subg_C=subg_C,
                ablation_innerGNN=model_config.ablation_innerGNN,
                ablation_PE=model_config.ablation_PE,
                extend_r=model_config.extend_r,
                predict_force=False
            )
        
        # ema configs
        self.ema_model = ExponentialMovingAverage(self, decay=optimizer_config.ema_decay, device=self.device)
        

        # optimizer and scheduler configs
        self.automatic_optimization = False
        
        self.optimizer_config = optimizer_config
        self.scheduler_config = scheduler_config
        
        self.force_ratio = optimizer_config.force_ratio
        self.gradient_clip_val = optimizer_config.gradient_clip_val
        self.warmup_end = False
        self.warmup_epoch = scheduler_config.warmup_epoch
        
        

    def forward(self, batch_data):
        return self.model(batch_data)
        
    def configure_optimizers(self):
        return configure_optimizers_(self)

    def training_step(self, batch, batch_idx):
        batch_size = batch.z.shape[0]
        self.train()
        pred_y = self(batch).squeeze()
        
        pred_y = pred_y.view(batch.y.shape)
        loss = torch.nn.functional.mse_loss(pred_y, batch.y)
        mae = torch.nn.functional.l1_loss(pred_y, batch.y)
        # log multiple loss
        
        self.log('train_loss/train_loss', loss, prog_bar=True, on_epoch=True, on_step=False, sync_dist=True, batch_size=batch_size)
        self.log('train_loss/train_mae', mae, prog_bar=True, on_epoch=True, on_step=False, sync_dist=True, batch_size=batch_size)
            
        # optimize manually
        self.opt_step(loss)
        
        # log
        self.log("learning_rate/lr_rate_AdamW", self.optimizers().param_groups[0]["lr"], on_epoch=True, on_step=False, sync_dist=True, batch_size=batch_size)
        
        return loss
        
        
    def validation_step(self, batch, batch_idx):
        batch_size = batch.z.shape[0]
        self.eval()
        pred_y = self(batch).squeeze() # (B, 1)
        
        pred_y = pred_y.view(batch.y.shape) # (B, 1)
        bs, _ = pred_y.shape
        pred_higher = pred_y[0::2, :] > pred_y[1::2, :]
        batch_higher = batch.y[0::2, :] > batch.y[1::2, :] 
        acc = (pred_higher == batch_higher).sum().item() / len(pred_higher)
        loss = torch.nn.functional.l1_loss(pred_y, batch.y)
    
        for i in range(0, bs, 2):
            assert batch.name[i][0] != batch.name[i+1][0] and batch.name[i][1] == batch.name[i+1][1]
        

        # log multiple loss
        self.log('val_loss/acc', acc, prog_bar=True, on_epoch=True, on_step=False, sync_dist=True, batch_size=batch_size)
        self.log('val_loss/val_loss', loss, prog_bar=True, on_epoch=True, on_step=False, sync_dist=True, batch_size=batch_size)

        return acc
        
        
    def test_step(self, batch, batch_idx):
        batch_size = batch.z.shape[0]
        self.eval()
        pred_y = self(batch).squeeze() # (B, 1)
        
        pred_y = pred_y.view(batch.y.shape) # (B, 1)
        bs, _ = pred_y.shape
        pred_higher = pred_y[0::2, :] > pred_y[1::2, :]
        batch_higher = batch.y[0::2, :] > batch.y[1::2, :] 
        acc = (pred_higher == batch_higher).sum().item() / len(pred_higher)
        
        for i in range(0, bs, 2):
            assert batch.name[i][0] != batch.name[i+1][0] and batch.name[i][1] == batch.name[i+1][1]
        
        self.log(f'test_loss/acc', acc, prog_bar=True, on_epoch=True, on_step=False, sync_dist=True, batch_size=batch_size)

        return acc

        
    def on_validation_epoch_end(self):
        if self.warmup_end:
            if type(self.lr_schedulers()) is list:
                for scheduler in self.lr_schedulers():
                    if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                        scheduler.step(self.trainer.callback_metrics[self.scheduler_config.monitor])
            else:
                if isinstance(self.lr_schedulers(), torch.optim.lr_scheduler.ReduceLROnPlateau):
                        self.lr_schedulers().step(self.trainer.callback_metrics[self.scheduler_config.monitor])
                        
                        
                        
        
    def opt_step(self, loss) -> None:
        if type(self.lr_schedulers()) is list:
            lr = self.lr_schedulers()[0].optimizer.param_groups[0]["lr"]
        else:
            lr = self.lr_schedulers().optimizer.param_groups[0]["lr"]
        self.optimizers().optimizer.param_groups[0]["lr"] = lr
        self.optimizers().param_groups[0]["lr"] = lr
        
        self.optimizers().zero_grad()
        self.manual_backward(loss)
        clip_grad_norm_(self.parameters(), self.gradient_clip_val)
        self.optimizers().step()
        
    def on_train_epoch_end(self):
        if self.warmup_end == False:
            if self.current_epoch >= self.warmup_epoch:
                self.warmup_end = True
        if type(self.lr_schedulers()) is list:
            for scheduler in self.lr_schedulers():
                if not isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    scheduler.step()
        else:
            if not isinstance(self.lr_schedulers(), torch.optim.lr_scheduler.ReduceLROnPlateau):
                self.lr_schedulers().step()
        self.trainer.train_dataloader.dataset.reshuffle_grouped_dataset()
    
    def on_train_batch_end(self, outputs, batch, batch_idx):
        self.ema_model.update_parameters(self) 