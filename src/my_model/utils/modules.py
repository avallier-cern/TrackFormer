import math
from torch import softmax, nn, optim
import torch
import numpy as np
from torch.nn.functional import mse_loss, l1_loss
import lightning as L
from abc import ABC, abstractmethod


################################### Helper functions:


def scaled_dot_product(q, k, v, mask=None):
    """
    Performs scaled dot-product attention.

    Args:
        q, k, v (torch.Tensor)  : Query , Key , value  tensors  (B, num_heads, seq_len, head_dim).
        mask : batch first mask
    """

    L, S = q.size(-2), k.size(-2)
    B, num_heads = q.size(0), q.size(1)

    # [Batch, NumHeads, SeqLen, SeqLen]
    attn_bias = torch.zeros(B, num_heads, L, S, dtype=q.dtype, device=q.device)

    scale_factor = 1 / math.sqrt(q.size(-1))

    # Make sure that the mask is broadcastable
    if mask is not None:
        mask = mask.unsqueeze(1).unsqueeze(1)  # [Batch, 1, 1, SeqLen]
        attn_bias.masked_fill_(mask.logical_not(), float("-inf"))

    attn_weight = q @ k.transpose(-2, -1) * scale_factor
    attn_weight += attn_bias

    attention = torch.softmax(attn_weight, dim=-1)

    if mask is not None:
        attention = attention.permute(0, 1, 3, 2).masked_fill(mask == 0, 0)
        attention = attention.permute(0, 1, 3, 2)

    values = attention @ v
    return values, attention


################################### Positional Encoding for RoPE:


class RotaryPositionalEncoding(nn.Module):
    def __init__(self, model_dim):
        super().__init__()
        self.model_dim = model_dim
        theta = 10000 ** (-torch.arange(0, model_dim, 2).float() / model_dim)
        self.register_buffer("theta", theta)

    def forward(self, q, k):
        seq_len = q.shape[2]
        theta = self.theta[: self.model_dim // 2].unsqueeze(0).unsqueeze(0)
        m = torch.arange(seq_len, device=q.device).float().unsqueeze(1) * theta
        cos_m, sin_m = torch.cos(m), torch.sin(m)

        q1, q2 = q[..., 0::2], q[..., 1::2]
        k1, k2 = k[..., 0::2], k[..., 1::2]

        q_rot = torch.cat([q1 * cos_m - q2 * sin_m, q1 * sin_m + q2 * cos_m], dim=-1)
        k_rot = torch.cat([k1 * cos_m - k2 * sin_m, k1 * sin_m + k2 * cos_m], dim=-1)

        return q_rot, k_rot


#################################### TrackFormer layers:


class MultiheadAttention(nn.Module):
    """
    Multihead attention mechanism.
    ------------------------------

    Args:
        input_dim (int): Input dimension.
        embed_dim (int): Embedding dimension.
        num_heads (int): Number of attention heads.
    """

    def __init__(self, input_dim, embed_dim, num_heads, use_rope=False):
        super().__init__()
        assert (
            embed_dim % num_heads == 0
        ), "Embedding dimension must be divisible among heads"

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads  # d_k
        self.use_rope = use_rope

        self.qkv_proj = nn.Linear(input_dim, 3 * embed_dim)  # stacked matrices
        self.o_proj = nn.Linear(embed_dim, embed_dim)
        self.rope = RotaryPositionalEncoding(self.head_dim) if use_rope else None

        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.qkv_proj.weight)
        self.qkv_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.o_proj.weight)
        self.o_proj.bias.data.fill_(0)

    def forward(self, x, mask=None, return_attention=False):
        batch_size, seq_length, _ = x.size()
        qkv = self.qkv_proj(x)

        # Separate Q, K, V
        qkv = qkv.reshape(batch_size, seq_length, self.num_heads, 3 * self.head_dim)
        qkv = qkv.permute(0, 2, 1, 3)  # [B, Head, SeqLen, Dims]
        q, k, v = qkv.chunk(3, dim=-1)

        if self.use_rope:
            q, k = self.rope(q, k)

        values, attention = scaled_dot_product(q, k, v, mask=mask)
        values = values.permute(0, 2, 1, 3)  # [B, SeqLen, Head, Dims]
        values = values.reshape(batch_size, seq_length, self.embed_dim)
        o = self.o_proj(values)

        return (o, attention) if return_attention else o


class EncoderBlock(nn.Module):
    """
    Transformer encoder block.

    Args:
        input_dim (int): Input dimension.
        num_heads (int): Number of attention heads.
        dim_feedforward (int): Dimension of the feedforward network.
        dropout (float, optional): Dropout rate. Defaults to 0.0.
    """

    def __init__(
        self, input_dim, num_heads, dim_feedforward, dropout=0.0, use_rope=False
    ):
        super().__init__()

        # Attention
        self.self_attn = MultiheadAttention(input_dim, input_dim, num_heads, use_rope)

        # Feedforward
        self.linear_net = nn.Sequential(
            nn.Linear(input_dim, dim_feedforward),
            nn.Dropout(dropout),
            nn.LeakyReLU(inplace=True),
            nn.Linear(dim_feedforward, input_dim),
        )

        # Layers Norms
        self.norm1 = nn.LayerNorm(input_dim)
        self.norm2 = nn.LayerNorm(input_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # Attention
        attn_out = self.self_attn(x, mask=mask)
        x = x + self.dropout(attn_out)
        x = self.norm1(x)

        # FeedForward
        linear_out = self.linear_net(x)
        x = x + self.dropout(linear_out)
        x = self.norm2(x)

        return x


class TransformerEncoder(nn.Module):

    def __init__(self, num_layers, **block_args):
        super().__init__()
        self.layers = nn.ModuleList(
            [EncoderBlock(**block_args) for _ in range(num_layers)]
        )

    def forward(self, x, mask=None):
        for layer in self.layers:
            x = layer(x, mask=mask)
        return x

    def get_attention_maps(self, x, mask=None):
        attention_maps = []
        for layer in self.layers:
            _, attn_map = layer.self_attn(x, mask=mask, return_attention=True)
            attention_maps.append(attn_map)
            x = layer(x)
        return attention_maps


class CosineWarmupScheduler(optim.lr_scheduler._LRScheduler):

    def __init__(self, optimizer, warmup, max_iters, min_lr=0.0):
        self.warmup = warmup
        self.max_iters = max_iters
        self.min_lr = min_lr
        # if hasattr(trainer.train_dataloader, '__len__'):
        #     self.max_num_iters = trainer.max_epochs * len(trainer.train_dataloader)
        # else:
        #     self.max_num_iters = trainer.max_epochs * trainer.limit_train_batches
        super().__init__(optimizer)

    def get_lr(self):
        if self.last_epoch > self.max_iters:
            return [self.min_lr for _ in self.base_lrs]
        lr_factor = self.get_lr_factor(epoch=self.last_epoch)
        return [max(base_lr * lr_factor, self.min_lr) for base_lr in self.base_lrs]

    def get_lr_factor(self, epoch):
        lr_factor = 0.5 * (1 + np.cos(np.pi * epoch / self.max_iters))
        if epoch <= self.warmup:
            lr_factor *= epoch * 1.0 / self.warmup
        return lr_factor


class Loss:
    def __init__(self, mode="mse"):
        super().__init__()
        self.mode = mode
        self.quantile = None

        if "qloss" in self.mode:
            _, q = self.mode.split("-")
            self.quantile = float(q)
            self.loss_fn = self._quantile_loss
        elif "mse" == self.mode:
            self.loss_fn = mse_loss
        elif "mse_angle" == self.mode:
            # Use 2*(1-cos(theta)) instead of angle directly
            # https://stats.stackexchange.com/a/565057
            # https://stats.stackexchange.com/a/425270
            self.loss_fn = lambda preds, targets: torch.mean(
                (2 * (1 - torch.cos(preds - targets)))
            )
        elif "mae" == self.mode:
            self.loss_fn = l1_loss
        elif "mse_inv" == self.mode:
            eps = self.mode.split("_")[-1]
            eps = float(eps)
            # EPS and absolute value are used to avoid division by zero
            self.loss_fn = lambda preds, targets: mse_loss(
                torch.sign(preds) * torch.abs(1 / (preds + eps)),
                torch.sign(targets) * torch.abs(1 / (targets + eps)),
            ) + mse_loss(preds, targets)
        elif "rel_mse" == self.mode:
            self.loss_fn = lambda preds, targets: torch.mean(
                torch.square((preds - targets) / targets)
            )
        elif "rel_rmse_percent" == self.mode:
            self.loss_fn = (
                lambda preds, targets: torch.sqrt(
                    torch.mean(torch.square((preds - targets) / targets))
                )
                * 100
            )
        elif "gaussian_nll" == self.mode:
            self.loss_fn = nn.GaussianNLLLoss()
        else:
            raise ValueError(f"Uknown loss funtion: {self.mode}")

    def _quantile_loss(self, preds, targets):
        errors = targets - preds
        return torch.mean(
            torch.max((self.quantile - 1) * errors, self.quantile * errors)
        )

    def __call__(self, preds, targets, vars=None):
        if "gaussian_nll" == self.mode:
            print("preds at 276 (beginning) are:", preds)
            print("preds shape at 276 (beginning) are:", preds.shape)
            return self.loss_fn(preds, targets, vars)
        else:
            return self.loss_fn(preds, targets)


class Metric:
    def __init__(self, mode="mse"):
        super().__init__()
        self.mode = mode

        if "mse" == self.mode:
            self.metric_fn = mse_loss
        elif "mae" == self.mode:
            self.metric_fn = l1_loss
        elif "sign" == self.mode:
            self.metric_fn = lambda preds, targets: torch.mean(
                (torch.sign(preds) == torch.sign(targets)).float()
            )
        elif "resolution_bias" == self.mode:
            self.metric_fn = lambda preds, targets: torch.mean(
                (preds - targets) / targets
            )
        elif "resolution_std" == self.mode:
            self.metric_fn = lambda preds, targets: torch.std(
                (preds - targets) / targets
            )
        else:
            raise ValueError(f"Uknown metric funtion: {self.mode}")

    def __call__(self, preds, targets):
        return self.metric_fn(preds, targets)


class BaseModel(L.LightningModule):
    """
    Base LightningModule for training and evaluating models.
    Optimiser args:
        lr (float): Learning rate.
        warmup (int): Number of warmup steps, [50, 500].
        max_iters (int): Maximum number of iterations the model is trained for, used by the CosineWarmup scheduler.
    Loss args:
        loss_type (dic or str): Type of loss function 'mse', "mae' or 'qloss- q_value'.
        quantile (float, optional): Quantile value for quantile loss, if used. Default is 0.5.
    """

    def __init__(self):
        super().__init__()
        if isinstance(self.hparams.criterion, str):
            self.hparams.criterion = [self.hparams.criterion] * self.hparams.num_classes
        self.criterion = []
        for criterion in self.hparams.criterion:
            self.criterion.append(Loss(criterion))
        assert len(self.criterion) > 0, "At least one criterion must be specified"
        assert (
            len(self.criterion) == self.hparams.num_classes
        ), "Number of criteria must be str or match the number of classes"
        self.metric = Metric(self.hparams.metric) if self.hparams.metric else None
        self._loss_param_names = None

    def _sanitize_metric_token(self, name):
        token = str(name).strip()
        if not token:
            return ""
        return token.replace(" ", "_").replace("/", "_")

    def _resolve_loss_param_names(self):
        names = (
            getattr(self.hparams, "output_variables", None)
            or getattr(self.hparams, "output_names", None)
            or getattr(self.hparams, "target_names", None)
        )

        if names is None and getattr(self, "trainer", None) is not None:
            dm = getattr(self.trainer, "datamodule", None)
            if dm is not None:
                names = getattr(dm, "output_variables", None)
                if names is None and hasattr(dm, "hparams"):
                    names = getattr(dm.hparams, "output_variables", None)
                    if names is None:
                        kwargs = getattr(dm.hparams, "kwargs", None)
                        if isinstance(kwargs, dict):
                            names = kwargs.get("output_variables", None)

        if names is None:
            return []

        if isinstance(names, str):
            names = [names]
        elif not isinstance(names, (list, tuple)):
            return []

        return [self._sanitize_metric_token(name) for name in names]

    def _get_loss_param_token(self, index):
        if self._loss_param_names is None:
            self._loss_param_names = self._resolve_loss_param_names()

        if index < len(self._loss_param_names) and self._loss_param_names[index]:
            return self._loss_param_names[index]

        return f"param_{index}"

    def setup(self, stage=None):
        if stage == "fit" and self.trainer.datamodule:
            # Get the total number of steps in the dataset
            if hasattr(self.trainer.datamodule.train_dataloader(), "__len__"):
                self.total_steps = len(self.trainer.datamodule.train_dataloader())
            else:
                self.total_steps = sum(
                    1 for _ in self.trainer.datamodule.train_dataloader()
                )
            print(f"Total steps in dataset: {self.total_steps}")

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.hparams.lr)
        if self.hparams.use_scheduler:
            self.max_cosine_iters = self.total_steps * min(
                1000, self.trainer.max_epochs
            )
            lr_scheduler = CosineWarmupScheduler(
                optimizer,
                warmup=self.hparams.warmup,
                max_iters=self.max_cosine_iters,
                min_lr=self.hparams.min_lr,
            )
            return [optimizer], [{"scheduler": lr_scheduler, "interval": "step"}]
        return optimizer

    def _log_grouped_scalar(self, tag, mode, value, step=None):
        if self.logger is None:
            return

        experiment = getattr(self.logger, "experiment", None)
        if experiment is None:
            return

        scalar_value = value.detach().item() if torch.is_tensor(value) else value
        log_step = self.global_step if step is None else int(step)

        if hasattr(experiment, "add_scalars"):
            experiment.add_scalars(tag, {mode: scalar_value}, log_step)
            return

        if hasattr(experiment, "add_scalar"):
            experiment.add_scalar(f"{tag}/{mode}", scalar_value, log_step)
            return

        if hasattr(experiment, "log"):
            experiment.log({f"{tag}/{mode}": scalar_value}, step=log_step)

    def _calculate_loss(self, batch, mode="train"):

        inputs, mask, label = batch

        preds = self(inputs, mask=mask)
        print("preds at 436 are:", preds)
        print("preds at 436 shape is:", preds.shape)
        # Separate loss for each parameter
        losses = []

        print("Criterion length:",len(self.criterion))

        # if len(self.criterion) == 5: # case without uncertainties
        #     for i, criterion in enumerate(self.criterion):
        #         preds_i = preds[:, i].squeeze()
        #         label_i = label[:, i].squeeze()
        #         param_token = self._get_loss_param_token(i)
        #         # Optional normalization of the loss
        #         if (
        #             hasattr(self.hparams, "norm_loss")
        #             and self.hparams.norm_loss is not None
        #         ):
        #             if self.hparams.norm_loss == "std":
        #                 preds_i = preds_i / torch.std(label_i)
        #                 label_i = label_i / torch.std(label_i)
        #             else:
        #                 raise ValueError(
        #                     f"Unknown norm_loss method: {self.hparams.norm_loss}"
        #                 )
        #         loss = criterion(preds_i, label_i)
        #         losses.append(loss)
        #         self.log(
        #             f"{mode}_loss_{param_token}",
        #             loss,
        #             prog_bar=True,
        #             logger=False,
        #             batch_size=inputs.shape[0],
        #         ) # old version
        if len(self.criterion) == 5: 
            for i, criterion in enumerate(self.criterion):
                if i < 5:
                    preds_i = preds[:, i].squeeze()
                    print("preds_i is:", preds_i)
                    print("preds_i shape is:", preds_i.shape)
                    label_i = label[:, i].squeeze()
                    vars_i = t2 = torch.ones(preds_i.shape, device='cuda:0')
                    param_token = self._get_loss_param_token(i)
                    # Optional normalization of the loss
                    if (
                        hasattr(self.hparams, "norm_loss")
                        and self.hparams.norm_loss is not None
                    ):
                        if self.hparams.norm_loss == "std":
                            preds_i = preds_i / torch.std(label_i)
                            label_i = label_i / torch.std(label_i)
                        else:
                            raise ValueError(
                                f"Unknown norm_loss method: {self.hparams.norm_loss}"
                            )
                    loss = criterion(preds_i, label_i, vars_i)
                    print("Loss is:",loss)
                    losses.append(loss)
                    self.log(
                        f"{mode}_loss_{param_token}",
                        loss,
                        prog_bar=True,
                        logger=False,
                        batch_size=inputs.shape[0],
                    )
                else:
                    break
        else:
            raise ValueError(
                f"Number of criterion can only be 5 or 10"
            )
        
        # Total loss
        if self.hparams.aggregate_loss == "sum":
            loss = torch.sum(torch.stack(losses))
        elif self.hparams.aggregate_loss == "mean":
            loss = torch.mean(torch.stack(losses))
        elif self.hparams.aggregate_loss == "geometric_mean":
            loss = torch.prod(torch.stack(losses)) ** (1.0 / len(losses))
        else:
            raise ValueError(
                f"Unknown aggregate loss method: {self.hparams.aggregate_loss}"
            )
        self.log(
            f"{mode}_loss",
            loss,
            prog_bar=True,
            logger=False,
            batch_size=inputs.shape[0],
        )
        # Access to log_every_n_steps
        log_every_n_steps = self.trainer.log_every_n_steps

        if self.logger and (
            self.global_step % log_every_n_steps == 0 or mode != "train"
        ):
            # Log total loss
            self._log_grouped_scalar("loss", mode, loss)
            # Add individual losses
            for i, l in enumerate(losses):
                param_tag = f"loss_{self._get_loss_param_token(i)}"
                self._log_grouped_scalar(
                    param_tag, mode, l
                )

        # Early return
        if self.metric is None:
            return loss

        # Calculate metric
        metric = self.metric(preds.squeeze(), label.squeeze())
        self.log(
            f"{mode}_{self.metric.mode}_metric",
            metric,
            prog_bar=True,
            logger=False,
            batch_size=inputs.shape[0],
        )

        if self.logger and (
            self.global_step % log_every_n_steps == 0 or mode != "train"
        ):
            self._log_grouped_scalar(f"{self.metric.mode}_metric", mode, metric)
        return loss

    def training_step(self, batch, batch_idx):
        return self._calculate_loss(batch, mode="train")

    def validation_step(self, batch, batch_idx):
        _ = self._calculate_loss(batch, mode="val")

    def test_step(self, batch, batch_idx):
        _ = self._calculate_loss(batch, mode="test")
