import math
import os
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import scipy as sc
import torch
import torch.nn as nn

from data_provider.data_factory import get_data
from exp.exp_basic import Exp_Basic
from models.model_factory import get_model
from utils.drag_coefficient import cal_coefficient
from utils.loss import L2Loss, DerivLoss
from utils.visual import visual

try:
    from lion_pytorch import Lion
except ImportError:
    Lion = None


class Exp_Steady_Design(Exp_Basic):
    def __init__(self, args):
        super(Exp_Steady_Design, self).__init__(args)
        self._abupt_mode = getattr(self.args, 'model', '') == 'AB_UPT'
        self._abupt_loss_fn = nn.MSELoss()

    def vali(self):
        if self._abupt_mode:
            return self._abupt_vali()
        myloss = nn.MSELoss(reduction='none')
        self.model.eval()
        rel_err = 0.0
        index = 0
        with torch.no_grad():
            for pos, fx, y, surf, geo, obj_file in self.test_loader:
                x, fx, y, geo = pos.cuda(), fx.cuda(), y.cuda(), geo.cuda()
                if self.args.fun_dim == 0:
                    fx = None
                with self._maybe_autocast():
                    out = self.model(x.unsqueeze(0), fx.unsqueeze(0), geo=geo)[0]
                loss_press = myloss(out[surf, -1], y[surf, -1]).mean(dim=0)
                loss_velo_var = myloss(out[:, :-1], y[:, :-1]).mean(dim=0)
                loss_velo = loss_velo_var.mean()
                loss = loss_velo + 0.5 * loss_press
                rel_err += loss.item()
                index += 1

        rel_err /= float(index)
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
                use_triton=True,
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
        else:
            scheduler = None
        myloss = nn.MSELoss(reduction='none')

        for ep in range(self.args.epochs):

            self.model.train()
            train_loss = 0
            index = 0
            if self._abupt_mode:
                iter_loader = self.train_loader
            else:
                iter_loader = self.train_loader
            for batch in iter_loader:
                if self._abupt_mode:
                    batch = self._move_abupt_batch_to_device(batch)
                    self._zero_grad(optimizer)
                    with self._maybe_autocast():
                        outputs = self.model(batch)
                        loss = self._abupt_loss(outputs, batch)
                    train_loss += loss.item()
                    index += 1
                    self._backward(loss, optimizer)
                    self._optimizer_step(optimizer)
                    continue

                pos, fx, y, surf, geo = batch
                x, fx, y, geo = pos.cuda(), fx.cuda(), y.cuda(), geo.cuda()
                if self.args.fun_dim == 0:
                    fx = None
                self._zero_grad(optimizer)
                with self._maybe_autocast():
                    out = self.model(x.unsqueeze(0), fx.unsqueeze(0), geo=geo)[0]
                    loss_press = myloss(out[surf, -1], y[surf, -1]).mean(dim=0)
                    loss_velo_var = myloss(out[:, :-1], y[:, :-1]).mean(dim=0)
                    loss_velo = loss_velo_var.mean()
                    loss = loss_velo + 0.5 * loss_press

                train_loss += loss.item()
                index += 1
                self._backward(loss, optimizer)
                self._optimizer_step(optimizer)

                if self.args.scheduler in ['OneCycleLR', 'WarmupCosine'] and scheduler is not None:
                    scheduler.step()
            if self.args.scheduler in ['CosineAnnealingLR', 'StepLR'] and scheduler is not None:
                scheduler.step()

            train_loss = train_loss / float(index)
            print("Epoch {} Train loss : {:.5f}".format(ep, train_loss))

            rel_err = self.vali()
            print("rel_err:{}".format(rel_err))

            lr = optimizer.param_groups[0]['lr']
            self.log_metrics({'train/loss': train_loss, 'val/loss': rel_err, 'train/lr': lr})

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
        if self._abupt_mode:
            return self._abupt_test()
        self.model.load_state_dict(torch.load("./checkpoints/" + self.args.save_name + ".pt"))
        self.model.eval()
        if not os.path.exists('./results/' + self.args.save_name + '/'):
            os.makedirs('./results/' + self.args.save_name + '/')

        export_path = getattr(self.args, 'export_surface_ply', None)
        export_enabled = bool(export_path)
        export_limit = max(1, int(getattr(self.args, 'export_surface_limit', 1)))
        export_sample_filter = getattr(self.args, 'export_surface_sample', None)
        export_include_error = bool(getattr(self.args, 'export_surface_include_error', 1))
        export_records = []

        criterion_func = nn.MSELoss(reduction='none')
        l2errs_press = []
        l2errs_velo = []
        mses_press = []
        mses_velo_var = []
        times = []
        gt_coef_list = []
        pred_coef_list = []
        coef_error = 0
        index = 0
        with torch.no_grad():
            for pos, fx, y, surf, geo, obj_file in self.test_loader:
                pos_cpu = pos
                surf_cpu = surf
                x, fx, y, geo = pos.cuda(), fx.cuda(), y.cuda(), geo.cuda()
                if self.args.fun_dim == 0:
                    fx = None
                tic = time.time()
                with self._maybe_autocast():
                    out = self.model(x.unsqueeze(0), fx.unsqueeze(0), geo=geo)[0]
                toc = time.time()

                if self.test_loader.coef_norm is not None:
                    mean = torch.tensor(self.test_loader.coef_norm[2]).cuda()
                    std = torch.tensor(self.test_loader.coef_norm[3]).cuda()
                    pred_press = out[surf, -1] * std[-1] + mean[-1]
                    gt_press = y[surf, -1] * std[-1] + mean[-1]
                    pred_surf_velo = out[surf, :-1] * std[:-1] + mean[:-1]
                    gt_surf_velo = y[surf, :-1] * std[:-1] + mean[:-1]
                    pred_velo = out[~surf, :-1] * std[:-1] + mean[:-1]
                    gt_velo = y[~surf, :-1] * std[:-1] + mean[:-1]
                else:
                    pred_press = out[surf, -1]
                    gt_press = y[surf, -1]
                    pred_surf_velo = out[surf, :-1]
                    gt_surf_velo = y[surf, :-1]
                    pred_velo = out[~surf, :-1]
                    gt_velo = y[~surf, :-1]

                pred_coef = cal_coefficient(obj_file.split('/')[1], pred_press[:, None].detach().cpu().numpy(),
                                            pred_surf_velo.detach().cpu().numpy())
                gt_coef = cal_coefficient(obj_file.split('/')[1], gt_press[:, None].detach().cpu().numpy(),
                                          gt_surf_velo.detach().cpu().numpy())

                gt_coef_list.append(gt_coef)
                pred_coef_list.append(pred_coef)
                coef_error += (abs(pred_coef - gt_coef) / gt_coef)

                l2err_press = torch.norm(pred_press - gt_press) / torch.norm(gt_press)
                l2err_velo = torch.norm(pred_velo - gt_velo) / torch.norm(gt_velo)

                mse_press = criterion_func(out[surf, -1], y[surf, -1]).mean(dim=0)
                mse_velo_var = criterion_func(out[~surf, :-1], y[~surf, :-1]).mean(dim=0)

                l2errs_press.append(l2err_press.cpu().numpy())
                l2errs_velo.append(l2err_velo.cpu().numpy())
                mses_press.append(mse_press.cpu().numpy())
                mses_velo_var.append(mse_velo_var.cpu().numpy())
                times.append(toc - tic)
                index += 1

                if export_enabled:
                    sample_id = self._resolve_sample_identifier(obj_file, index)
                    matches_filter = (export_sample_filter is None) or (export_sample_filter in sample_id)
                    under_limit = len(export_records) < export_limit or (export_sample_filter is not None and matches_filter)
                    if matches_filter and under_limit:
                        export_records.append(
                            self._prepare_surface_export_record(
                                sample_id,
                                pos_cpu,
                                surf_cpu,
                                pred_press,
                                gt_press,
                            )
                        )

        gt_coef_list = np.array(gt_coef_list)
        pred_coef_list = np.array(pred_coef_list)
        spear = sc.stats.spearmanr(gt_coef_list, pred_coef_list)[0]
        print("rho_d: ", spear)
        print("c_d: ", coef_error / index)
        l2err_press = np.mean(l2errs_press)
        l2err_velo = np.mean(l2errs_velo)
        rmse_press = np.sqrt(np.mean(mses_press))
        rmse_velo_var = np.sqrt(np.mean(mses_velo_var, axis=0))
        if self.test_loader.coef_norm is not None:
            rmse_press *= self.test_loader.coef_norm[3][-1]
            rmse_velo_var *= self.test_loader.coef_norm[3][:-1]
        print('relative l2 error press:', l2err_press)
        print('relative l2 error velo:', l2err_velo)
        print('press:', rmse_press)
        print('velo:', rmse_velo_var, np.sqrt(np.mean(np.square(rmse_velo_var))))
        print('time:', np.mean(times))

        metrics = {
            'test/rho_d': float(spear),
            'test/c_d': float(coef_error / index),
            'test/l2_press': float(l2err_press),
            'test/l2_velo': float(l2err_velo),
            'test/rmse_press': float(rmse_press),
            'test/rmse_velo_mean': float(np.sqrt(np.mean(np.square(rmse_velo_var)))),
            'test/mean_inference_time': float(np.mean(times)),
        }
        metrics.update({f'test/rmse_velo_{i}': float(val) for i, val in enumerate(np.atleast_1d(rmse_velo_var))})
        self.log_metrics(metrics)

        if export_records:
            self._write_surface_ply_exports(export_records, export_include_error)

        self.finish_logging()

    @staticmethod
    def _resolve_sample_identifier(obj_file, fallback_idx):
        if isinstance(obj_file, Path):
            return obj_file.as_posix()
        if isinstance(obj_file, str):
            return obj_file
        if isinstance(obj_file, (list, tuple)) and obj_file:
            return str(obj_file[0])
        return f"sample_{fallback_idx:04d}"

    def _prepare_surface_export_record(self, sample_id, pos_cpu, surf_cpu, pred_press, gt_press):
        surf_mask = surf_cpu.bool()
        surface_positions = pos_cpu[surf_mask]
        if surface_positions.numel() == 0:
            raise ValueError(f"No surface points available for sample {sample_id}")
        surface_positions = surface_positions.detach().cpu().numpy()
        pred_vals = pred_press.detach().cpu().numpy().reshape(-1)
        gt_vals = gt_press.detach().cpu().numpy().reshape(-1)
        if surface_positions.shape[0] != pred_vals.shape[0]:
            raise ValueError(
                f"Surface vertex count mismatch for {sample_id}: "
                f"{surface_positions.shape[0]} positions vs {pred_vals.shape[0]} predictions"
            )
        return {
            'sample_name': sample_id,
            'positions': surface_positions.astype(np.float32),
            'pressure_pred': pred_vals.astype(np.float32),
            'pressure_gt': gt_vals.astype(np.float32),
        }

    def _write_surface_ply_exports(self, exports, include_error):
        base_path = Path(self.args.export_surface_ply)
        single_file = base_path.suffix.lower() == '.ply' and len(exports) == 1
        if single_file:
            targets = [(base_path, exports[0])]
            base_path.parent.mkdir(parents=True, exist_ok=True)
        else:
            export_dir = base_path if base_path.suffix.lower() != '.ply' else base_path.parent
            export_dir.mkdir(parents=True, exist_ok=True)
            targets = []
            for record in exports:
                safe_name = record['sample_name'].replace('/', '_')
                targets.append((export_dir / f"{safe_name}.ply", record))

        for path, record in targets:
            self._write_single_surface_ply(path, record, include_error)
            print(f"Saved surface predictions to {path}")

    @staticmethod
    def _write_single_surface_ply(path, record, include_error):
        positions = record['positions']
        pred = record['pressure_pred']
        gt = record['pressure_gt']
        num_vertices = positions.shape[0]
        if positions.shape[1] != 3:
            raise ValueError(f"Expected Nx3 positions, got shape {positions.shape}")
        if pred.shape[0] != num_vertices or gt.shape[0] != num_vertices:
            raise ValueError("Vertex/value count mismatch while writing PLY.")

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('w', encoding='utf-8') as fh:
            fh.write("ply\n")
            fh.write("format ascii 1.0\n")
            fh.write(f"element vertex {num_vertices}\n")
            fh.write("property float x\nproperty float y\nproperty float z\n")
            fh.write("property float pressure_pred\n")
            fh.write("property float pressure_gt\n")
            if include_error:
                fh.write("property float pressure_error\n")
            fh.write("end_header\n")
            for idx in range(num_vertices):
                x, y, z = positions[idx]
                pred_val = float(pred[idx])
                gt_val = float(gt[idx])
                if include_error:
                    err = pred_val - gt_val
                    fh.write(f"{x:.6f} {y:.6f} {z:.6f} {pred_val:.6f} {gt_val:.6f} {err:.6f}\n")
                else:
                    fh.write(f"{x:.6f} {y:.6f} {z:.6f} {pred_val:.6f} {gt_val:.6f}\n")
