import math
import os
import torch
from exp.exp_basic import Exp_Basic
from models.model_factory import get_model
from data_provider.data_factory import get_data
from utils.loss import L2Loss, DerivLoss
import matplotlib.pyplot as plt
from utils.visual import visual
import numpy as np

try:
    from lion_pytorch import Lion
except ImportError:
    Lion = None

class Exp_Steady(Exp_Basic):
    def __init__(self, args):
        super(Exp_Steady, self).__init__(args)

    def vali(self):
        myloss = L2Loss(size_average=False)
        self.model.eval()
        rel_err = 0.0
        with torch.no_grad():
            for pos, fx, y in self.test_loader:
                x, fx, y = pos.cuda(), fx.cuda(), y.cuda()
                if self.args.fun_dim == 0:
                    fx = None
                with self._maybe_autocast():
                    out = self.model(x, fx)
                if self.args.normalize:
                    out = self.dataset.y_normalizer.decode(out)

                tl = myloss(out, y).item()
                rel_err += tl

        rel_err /= self.args.ntest
        return rel_err

    def train(self):
        if self.args.optimizer == 'AdamW':
            optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay)
        elif self.args.optimizer == 'Adam':
            optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay)
        elif self.args.optimizer == 'Lion':
            if Lion is None:
                raise ImportError("lion_pytorch is required to use the Lion optimizer. Install it or choose another optimizer.")
            optimizer = Lion(
                self.model.parameters(),
                lr=self.args.lr,
                betas=(self.args.lion_beta1, self.args.lion_beta2),
                weight_decay=self.args.weight_decay,
            )
        else: 
            raise ValueError('Optimizer only AdamW, Adam, or Lion')
        
        if self.args.scheduler == 'OneCycleLR':
            scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=self.args.lr, epochs=self.args.epochs,
                                                            steps_per_epoch=len(self.train_loader),
                                                            pct_start=self.args.pct_start)
        elif self.args.scheduler == 'CosineAnnealingLR':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.args.epochs)
        elif self.args.scheduler == 'StepLR':
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=self.args.step_size, gamma=self.args.gamma)
        elif self.args.scheduler == 'WarmupCosine':
            steps_per_epoch = len(self.train_loader)
            if steps_per_epoch == 0:
                raise ValueError('Training DataLoader is empty; cannot configure scheduler.')
            total_steps = self.args.epochs * steps_per_epoch
            warmup_steps = max(1, int(total_steps * self.args.warmup_fraction))
            if warmup_steps >= total_steps:
                warmup_steps = max(total_steps - 1, 1)
            if self.args.min_lr > self.args.lr:
                raise ValueError('Argument min_lr must be less than or equal to lr.')
            min_lr_ratio = self.args.min_lr / self.args.lr if self.args.lr > 0 else 0.0

            def lr_lambda(step: int) -> float:
                step = step + 1  # LambdaLR uses zero-based steps
                if step <= warmup_steps:
                    return step / max(1, warmup_steps)
                progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
                cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
                return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
            initial_lr = self.args.lr * lr_lambda(-1)
            for param_group in optimizer.param_groups:
                param_group["lr"] = initial_lr
        else:
            scheduler = None
        myloss = L2Loss(size_average=False)
        if self.args.derivloss:
            regloss = DerivLoss(size_average=False, shapelist=self.args.shapelist)

        for ep in range(self.args.epochs):

            self.model.train()
            train_loss = 0

            for pos, fx, y in self.train_loader:
                x, fx, y = pos.cuda(), fx.cuda(), y.cuda()
                if self.args.fun_dim == 0:
                    fx = None
                self._zero_grad(optimizer)
                with self._maybe_autocast():
                    out = self.model(x, fx)
                    if self.args.normalize:
                        out = self.dataset.y_normalizer.decode(out)
                        y = self.dataset.y_normalizer.decode(y)

                    if self.args.derivloss:
                        loss = myloss(out, y) + 0.1 * regloss(out, y)
                    else:
                        loss = myloss(out, y)

                train_loss += loss.item()
                self._backward(loss, optimizer)
                self._optimizer_step(optimizer)
                
                if self.args.scheduler in ['OneCycleLR', 'WarmupCosine'] and scheduler is not None:
                    scheduler.step()
            if self.args.scheduler in ['CosineAnnealingLR', 'StepLR'] and scheduler is not None:
                scheduler.step()

            train_loss = train_loss / self.args.ntrain
            print("Epoch {} Train loss : {:.5f}".format(ep, train_loss))

            rel_err = self.vali()
            print("rel_err:{}".format(rel_err))

            lr = optimizer.param_groups[0]['lr']
            self.log_metrics({'train/loss': train_loss, 'val/rel_err': rel_err, 'train/lr': lr})

            if ep % 100 == 0:
                if not os.path.exists('./checkpoints'):
                    os.makedirs('./checkpoints')
                print('save models')
                torch.save(self.model.state_dict(), os.path.join('./checkpoints', self.args.save_name + '.pt'))

        if not os.path.exists('./checkpoints'):
            os.makedirs('./checkpoints')
        print('final save models')
        torch.save(self.model.state_dict(), os.path.join('./checkpoints', self.args.save_name + '.pt'))

    def test(self):
        self.model.load_state_dict(torch.load("./checkpoints/" + self.args.save_name + ".pt"))
        self.model.eval()
        if not os.path.exists('./results/' + self.args.save_name + '/'):
            os.makedirs('./results/' + self.args.save_name + '/')

        rel_err = 0.0
        id = 0
        myloss = L2Loss(size_average=False)
        with torch.no_grad():
            for pos, fx, y in self.test_loader:
                id += 1
                x, fx, y = pos.cuda(), fx.cuda(), y.cuda()
                if self.args.fun_dim == 0:
                    fx = None
                with self._maybe_autocast():
                    out = self.model(x, fx)
                if self.args.normalize:
                    out = self.dataset.y_normalizer.decode(out)
                tl = myloss(out, y).item()
                rel_err += tl
                if id < self.args.vis_num:
                    print('visual: ', id)
                    visual(x, y, out, self.args, id)

        rel_err /= self.args.ntest
        print("rel_err:{}".format(rel_err))
        self.log_metrics({'test/rel_err': rel_err})
        self.finish_logging()
