import os
import torch
from exp.exp_basic import Exp_Basic
from models.model_factory import get_model
from data_provider.data_factory import get_data
from utils.loss import L2Loss
import matplotlib.pyplot as plt
from utils.visual import visual
import numpy as np


class Exp_Dynamic_Autoregressive(Exp_Basic):
    def __init__(self, args):
        super(Exp_Dynamic_Autoregressive, self).__init__(args)

    def vali(self):
        myloss = L2Loss(size_average=False)
        test_l2_full = 0
        self.model.eval()
        with torch.no_grad():
            for x, fx, yy in self.test_loader:
                x, fx, yy = x.cuda(), fx.cuda(), yy.cuda()
                for t in range(self.args.T_out):
                    if self.args.fun_dim == 0:
                        fx = None
                    with self._maybe_autocast():
                        im = self.model(x, fx=fx)
                    if t == 0:
                        pred = im
                    else:
                        pred = torch.cat((pred, im), -1)
                    fx = torch.cat((fx[..., self.args.out_dim:], im), dim=-1)
                if self.args.normalize:
                    pred = self.dataset.y_normalizer.decode(pred)
                test_l2_full += myloss(pred.reshape(x.shape[0], -1), yy.reshape(x.shape[0], -1)).item()
        test_loss_full = test_l2_full / (self.args.ntest)
        return test_loss_full

    def train(self):
        if self.args.optimizer == 'AdamW':
            optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay)
        elif self.args.optimizer == 'Adam':
            optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.lr, weight_decay=self.args.weight_decay)
        else: 
            raise ValueError('Optimizer only AdamW or Adam')
        if self.args.scheduler == 'OneCycleLR':
            scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=self.args.lr, epochs=self.args.epochs,
                                                            steps_per_epoch=len(self.train_loader),
                                                            pct_start=self.args.pct_start)
        elif self.args.scheduler == 'CosineAnnealingLR':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.args.epochs)
        elif self.args.scheduler == 'StepLR':
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=self.args.step_size, gamma=self.args.gamma)
        else:
            scheduler = None

        myloss = L2Loss(size_average=False)

        for ep in range(self.args.epochs):
            self.model.train()
            train_l2_step = 0
            train_l2_full = 0

            for pos, fx, yy in self.train_loader:
                loss = 0
                x, fx, yy = pos.cuda(), fx.cuda(), yy.cuda()
                for t in range(self.args.T_out):
                    y = yy[..., self.args.out_dim * t:self.args.out_dim * (t + 1)]
                    if self.args.fun_dim == 0:
                        fx = None
                    with self._maybe_autocast():
                        im = self.model(x, fx=fx)
                    loss += myloss(im.reshape(x.shape[0], -1), y.reshape(x.shape[0], -1))
                    if t == 0:
                        pred = im
                    else:
                        pred = torch.cat((pred, im), -1)

                    if self.args.teacher_forcing:
                        fx = torch.cat((fx[..., self.args.out_dim:], y), dim=-1)
                    else:
                        fx = torch.cat((fx[..., self.args.out_dim:], im), dim=-1)

                train_l2_step += loss.item()
                train_l2_full += myloss(pred.reshape(x.shape[0], -1), yy.reshape(x.shape[0], -1)).item()
                self._zero_grad(optimizer)
                self._backward(loss, optimizer)
                self._optimizer_step(optimizer)

                if self.args.scheduler == 'OneCycleLR' and scheduler is not None:
                    scheduler.step()
            if self.args.scheduler in ['CosineAnnealingLR', 'StepLR'] and scheduler is not None:
                scheduler.step()

            train_loss_step = train_l2_step / (self.args.ntrain * float(self.args.T_out))
            train_loss_full = train_l2_full / (self.args.ntrain)
            print("Epoch {} Train loss step : {:.5f} Train loss full : {:.5f}".format(ep, train_loss_step,
                                                                                      train_loss_full))

            test_loss_full = self.vali()
            print("Epoch {} Test loss full : {:.5f}".format(ep, test_loss_full))

            lr = optimizer.param_groups[0]['lr']
            self.log_metrics({'train/loss_step': train_loss_step,
                              'train/loss_full': train_loss_full,
                              'val/loss_full': test_loss_full,
                              'train/lr': lr})

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
            for x, fx, yy in self.test_loader:
                id += 1
                x, fx, yy = x.cuda(), fx.cuda(), yy.cuda()
                for t in range(self.args.T_out):
                    if self.args.fun_dim == 0:
                        fx = None
                    with self._maybe_autocast():
                        im = self.model(x, fx=fx)
                    fx = torch.cat((fx[..., self.args.out_dim:], im), dim=-1)
                    if t == 0:
                        pred = im
                    else:
                        pred = torch.cat((pred, im), -1)
                if self.args.normalize:
                    pred = self.dataset.y_normalizer.decode(pred)
                rel_err += myloss(pred.reshape(x.shape[0], -1), yy.reshape(x.shape[0], -1)).item()
                if id < self.args.vis_num:
                    print('visual: ', id)
                    for t in range(self.args.T_out):
                        visual(x, yy[:, :, self.args.out_dim * t:self.args.out_dim * (t + 1)],
                               pred[:, :, self.args.out_dim * t:self.args.out_dim * (t + 1)], self.args,
                               str(id) + '_' + str(t))

        rel_err /= self.args.ntest
        print("rel_err:{}".format(rel_err))
        self.log_metrics({'test/rel_err': rel_err})
        self.finish_logging()
