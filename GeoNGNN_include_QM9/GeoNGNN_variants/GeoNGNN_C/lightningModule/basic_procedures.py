import torch
from pytorch_lightning import Callback
from utils.GradualWarmupScheduler import GradualWarmupScheduler



class basic_train_callback(Callback):

    def ffon_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        pl_module.ema_model.update_parameters(pl_module) 
        


    def ffon_train_epoch_end(self, trainer, pl_module):
        
        if pl_module.warmup_end == False:
            if pl_module.current_epoch >= pl_module.warmup_epoch:
                pl_module.warmup_end = True
            
        for scheduler in pl_module.lr_schedulers():
            if not isinstance(scheduler, torch.optim.lr_scheduler.CosineAnnealingLR):
                scheduler.step()

                lr = scheduler.get_last_lr()[0]
                pl_module.optimizers().optimizer.param_groups[0]["lr"] = lr
                pl_module.optimizers().param_groups[0]["lr"] = lr
                
        pl_module.trainer.train_dataloader.dataset.reshuffle_grouped_dataset()



    def ffon_validation_epoch_end(self, trainer, pl_module):
        if pl_module.warmup_end:
            for scheduler in pl_module.lr_schedulers():
                if isinstance(scheduler, torch.optim.lr_scheduler.CosineAnnealingLR):
                    scheduler.step(pl_module.trainer.callback_metrics["val_loss/val_loss"])
                    
                    lr = scheduler.optimizer.param_groups[0]["lr"]
                    pl_module.optimizers().optimizer.param_groups[0]["lr"] = lr
                    pl_module.optimizers().param_groups[0]["lr"] = lr
                    pl_module.lr_schedulers()[1].after_scheduler.optimizer.param_groups[0]["lr"] = lr
                    

            

            
                    
def configure_optimizers_(self):
    optimizer_config = self.optimizer_config
    scheduler_config = self.scheduler_config
    
    # initialize AdamW
    optimizer = torch.optim.AdamW(
        params=self.parameters(),
        lr=optimizer_config.learning_rate,
        weight_decay=optimizer_config.weight_decay,
        amsgrad=True,
        )


    # initialize schedulers
    if scheduler_config.scheduler_type == "cosine":
        COSINE = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, 
            T_max=scheduler_config.T_max, 
            eta_min=scheduler_config.eta_min
            )
        COSINE_warmup = GradualWarmupScheduler(
            optimizer=optimizer, 
            multiplier=1., 
            total_epoch=scheduler_config.warmup_epoch, 
            after_scheduler=COSINE
            )
        lr_scheduler_configs = {
            "scheduler": COSINE_warmup,
            "interval": "epoch",
        }
    else:
        RLROP = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 
            factor=scheduler_config.RLROP_factor,
            patience=scheduler_config.RLROP_patience,
            threshold=scheduler_config.RLROP_threshold,
            cooldown=scheduler_config.RLROP_cooldown
        )

        EXLR = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=scheduler_config.EXLR_gamma)
        EXLR_warmup = GradualWarmupScheduler(
            optimizer=optimizer, 
            multiplier=1., 
            total_epoch=scheduler_config.warmup_epoch, 
            after_scheduler=EXLR
            )

        lr_scheduler_configs = []
        for sched in [RLROP, EXLR_warmup]:
            lr_scheduler_config = {
                "scheduler": sched,
                "interval": "epoch",
                "monitor": "val_loss/val_loss"
            }
            lr_scheduler_configs.append(lr_scheduler_config)


    return [optimizer], lr_scheduler_configs
            
