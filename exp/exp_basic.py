import os
import torch
from contextlib import nullcontext
from models.model_factory import get_model
from data_provider.data_factory import get_data


def count_parameters(model):
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        params = parameter.numel()
        total_params += params
    print(f"Total Trainable Params: {total_params}")
    return total_params


class Exp_Basic(object):
    def __init__(self, args):
        self.dataset, self.train_loader, self.test_loader, args.shapelist = get_data(args)
        self.model = get_model(args).cuda()
        self.args = args
        print(self.args)
        print(self.model)
        param_count = count_parameters(self.model)
        self._wandb = None
        self._wandb_run = None
        self._compiled = False
        requested_amp = bool(getattr(self.args, 'amp', 0))
        if requested_amp and not torch.cuda.is_available():
            print("Automatic mixed precision requested, but CUDA is not available. Disabling AMP.")
        self._grad_scaler = None
        if torch.cuda.is_available() and requested_amp:
            self._grad_scaler = torch.cuda.amp.GradScaler()
        self._amp_enabled = self._grad_scaler is not None
        self._init_wandb()
        self.log_metrics({'model/num_parameters': param_count,
                          'model/mixed_precision': int(bool(self._grad_scaler) and self._grad_scaler.is_enabled())})
        self._compile_model_if_requested()

    def vali(self):
        pass

    def train(self):
        pass

    def test(self):
        pass

    def _compile_model_if_requested(self):
        if not bool(getattr(self.args, 'torch_compile', 0)):
            return
        if not hasattr(torch, 'compile'):
            print('torch.compile is not available in this PyTorch build.')
            self.log_metrics({'model/compiled': 0, 'model/compile_mode': 'unavailable'})
            return
        mode = getattr(self.args, 'torch_compile_mode', 'default')
        try:
            self.model = torch.compile(self.model, mode=mode)  # type: ignore[attr-defined]
            self._compiled = True
            print(f"Model compiled with torch.compile (mode={mode}).")
            self.log_metrics({'model/compiled': 1, 'model/compile_mode': mode})
        except Exception as exc:  # pragma: no cover - compilation may fail depending on env
            print(f"torch.compile failed: {exc}. Continuing without compilation.")
            self.log_metrics({'model/compiled': 0, 'model/compile_mode': mode})

    @staticmethod
    def _to_serializable(value):
        if isinstance(value, (int, float, bool, str)) or value is None:
            return value
        if isinstance(value, (list, tuple)):
            return [Exp_Basic._to_serializable(v) for v in value]
        if isinstance(value, dict):
            return {k: Exp_Basic._to_serializable(v) for k, v in value.items()}
        return str(value)

    def _init_wandb(self):
        try:
            import wandb  # type: ignore
        except ImportError:
            print("wandb not installed, skipping logging.")
            return

        config = {k: self._to_serializable(v) for k, v in vars(self.args).items()}
        try:
            self._wandb = wandb
            self._wandb_run = wandb.init(project='StandardBenchPDE',
                                         name=self.args.save_name,
                                         config=config)
        except Exception as exc:  # pragma: no cover - wandb failures shouldn't abort training
            print(f"wandb initialisation failed: {exc}")
            self._wandb = None
            self._wandb_run = None

    def log_metrics(self, metrics, step=None):
        if self._wandb_run is not None:
            self._wandb.log(metrics, step=step)

    def finish_logging(self):
        if self._wandb_run is not None:
            self._wandb_run.finish()
            self._wandb_run = None

    def _maybe_autocast(self):
        if self._amp_enabled and torch.cuda.is_available():
            return torch.cuda.amp.autocast(dtype=torch.float16)
        return nullcontext()

    def _zero_grad(self, optimizer):
        optimizer.zero_grad(set_to_none=True)

    def _backward(self, loss, optimizer):
        if self._grad_scaler is not None and self._grad_scaler.is_enabled():
            self._grad_scaler.scale(loss).backward()
        else:
            loss.backward()

    def _optimizer_step(self, optimizer):
        if self._grad_scaler is not None and self._grad_scaler.is_enabled():
            if self.args.max_grad_norm is not None:
                self._grad_scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)
            self._grad_scaler.step(optimizer)
            self._grad_scaler.update()
        else:
            if self.args.max_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)
            optimizer.step()
