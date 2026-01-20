
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Union, Optional
from abc import ABC, abstractmethod


# Base Loss Class
class BaseLoss(ABC, nn.Module):
  def __init__(self):
    super().__init__()
    self.name = self.__class__.__name__

  @abstractmethod
  def forward(self, pred, target):
    """
    计算loss

    Args:
      pred: (B, T, D) or (B, T) predictions
      target: (B, T, D) or (B, T) targets
    Returns:
      loss: scalar tensor or unreduced tensor
    """
    pass

  def get_config(self) -> Dict:
    """返回当前配置的字典"""
    return {}


# Standard Losses
class MSELoss(BaseLoss):
  """
  Config: 
    loss_type: "mse"
    reduction: "mean" | "sum" | "none"
  """
  def __init__(self, reduction="mean"):
    super().__init__()
    self.reduction = reduction
    self.mse_loss = nn.MSELoss(reduction=reduction)

  def forward(self, pred, target):
    return self.mse_loss(pred, target)

  def get_config(self):
    return {"reduction": self.reduction}
  
class L1Loss(BaseLoss):
  """
  Config:
    loss_type: "l1"
    reduction: "mean" | "sum" | "none"
  """
  def __init__(self, reduction="mean"):
    super().__init__()
    self.reduction = reduction

  def forward(self, pred, target):
    if self.reduction == "mean":
      return torch.mean(torch.abs(pred-target))
    elif self.reduction == "sum":
      return torch.sum(torch.abs(pred-target))
    else: 
      return torch.abs(pred-target)

  def get_config(self):
    return {"reduction": self.reduction}

class SmoothL1Loss(BaseLoss):
  """
  SmoothL1Loss（Huber Loss）
  Config: 
    loss_type: "smooth_l1"
    beta: float (deafult: 1.0)
  """
  def __init__(self, beta=1.0):
    super().__init__()
    self.beta = beta
    self.smooth_l1 = nn.SmoothL1Loss(beta=beta, reduction='mean')

  def forward(self, pred, target):
    return self.smooth_l1(pred, target)

  def get_config(self):
    return {"beta": self.beta}

class RankLoss01Range(BaseLoss):
  """
  Rank Loss（排序损失）
  约束最后一个维度（dim）上的相对高低关系，使模型输出在时间维度上的排序与标签一致
  适用于 0-1 区间的输出，对每个时间步的 136 个特征之间的相对高低关系进行约束
  """
  def __init__(self, **kwargs):
   super().__init__(**kwargs)
   self.gamma_min = kwargs.get('gamma_min', 0.05)
   self.gamma_max = kwargs.get('gamma_max', 0.3)

  def forward(self, scores, y_true):
    """
    :param scores: 模型输出值, shape = [batch, seq_len, dim]
    :param y_true: 标签值, shape = [batch, seq_len, dim]
    :return: Rank Loss（constant）
    """
    batch, seq_len, dim = scores.shape
    device = scores.device

    gamma = torch.std(y_true, dim=-1, keepdim=True)
    gamma = torch.clamp(gamma, self.gamma_min, self.gamma_max)

    # 对每个（batch, seq_len）在dim维度随机采样成对索引
    idx_i = torch.randint(0, dim, (batch, seq_len), device=device)
    idx_j = torch.randint(0, dim, (batch, seq_len), device=device)

    # 根据索引提取对应的值
    batch_idx = torch.arange(batch, device=device).view(-1, 1).expand(-1, seq_len)
    seq_idx = torch.arange(seq_len, device=device).view(-1, 1).expand(batch, -1)

    # 提取对应的值[batch, seq_len]
    s_i = scores[batch_idx, seq_idx, idx_i]
    s_j = scores[batch_idx, seq_idx, idx_j]
    y_i = y_true[batch_idx, seq_idx, idx_i]
    y_j = y_true[batch_idx, seq_idx, idx_j]

    y_ij = torch.where(
      y_i >= y_j,
      torch.tensor(1.0, device=device),
      torch.tensor(-1.0, device=device)
    )

    delta = (s_i - s_j) * y_ij
    loss_terms = F.softplus(gamma.squeeze(-1) * delta)

    rank_loss = loss_terms.mean()

    return rank_loss

# variance weighted loss
class VarianceWeightedLoss(BaseLoss):
  """
  基于方差的加权MSE损失
  对高方差维度赋予更高权重，使模型更关注变化较大的参数
  Config:
    'loss_type': "variance_weighted",
    'min_weight': 0.1, float 最小权重
    'max_weight': 10.0, float 最大权重
    compute_from_data: bool (default: True) 是否从数据计算权重
    weights_file: str (optional) 从文件加载预计算的权重
  """
  def __init__(
    self,
    min_weight=0.1,
    max_weight=10.0,
    compute_from_data=True,
    weights_file=None
  ):
    super().__init__()
    self.min_weight=min_weight
    self.max_weight=max_weight
    self.compute_from_data=compute_from_data
    self.weights_file=weights_file

    self.weights = None
    self.reduction = 'none' # 返回unreduced, 由外部处理

    if weights_file is not None:
      self._load_weights_from_file(weights_file)

  def _load_weights_from_file(self, weights_file):
    weights = np.load(weights_file)
    self.register_buffer('weights', torch.from_numpy(eights).float())

  def set_weights_from_data(self, targets):
    """
    Args:
      targets: (N, T, D) numpy array
    """
    dim_var = torch.var(torch.from_numpy(targets).float(), dim=(0, 1)).numpy()
    weights = np.sqrt(dim_var + 1e-8)
    weights = np.clip(weights, self.min_weight, self.max_weight)
    weights = weights / weights.mean()
    if hasattr(self, 'weights'):
      delattr(self, 'weights')
    self.register_buffer('weights', torch.from_numpy(weights.float()))

  def forward(self, pred, target):
    """
    计算加权MSE loss
    Args:
      pred: (B, T, D) predictions
      target: (B, T, D) targets
    Returns:
      loss: (B, T, D) unreduced weighted squared error
    """
    se = (pred - target) ** 2 # (B, T, D)
    if self.weights is not None:
      weights = self.weights.to(pred.device)
      se = se * weights.view(1,1,-1)
    return se

  def get_config(self):
    config = {
      "min_weight": self.min_weight,
      "max_weight": self.max_weight,
      "compute_from_data": self.compute_from_data
    }
    if self.weights_file is not None:
      config["weights_file"] = self.weights_file
    return config

# temporal smoothness loss
class TemporalSmoothLoss(BaseLoss):
  """
  时间平滑损失 - 惩罚相邻帧之间的剧烈变化

  Config:
    loss_type: "temporal_smooth"
    method: "diff1" | "diff2"
    weight: float 损失权重
  """
  def __init__(self, method="diff2", weight=1.0):
    super().__init__()
    self.method = method
    self.weight = weight

  def forward(self, pred, target=None):
    """
    计算时间平滑损失
    Args:
      pred: (B, T, D) predictions
      target: ignored
    Returns:
      loss: scalar tensor
    """
    if self.method == "diff1":
      diff = pred[:, 1:] - pred[:, :-1]
      loss = torch.mean(diff ** 2)
    elif self.method == "diff2":
      diff = pred[:, 2:] - 2 * pred[:, 1:-1] + pred[:, :-2]
      loss = torch.mean(diff ** 2)
    else:
      raise ValueError()
    return self.weight * loss

  def get_config(self):
    return {"method": self.method, "weight": self.weight}


# combinded Loss
class CombinedLoss(BaseLoss):
  """
  组合多个loss
  Config:
    loss_type: "combined"
    losses: List[dict]

  每个loss配置：
  {
    "type": "mse" | "" | "" | ... ,
    "weight": float, # 该loss权重
    "params": dict # loss特定参数（可选）
  }
  """
  LOSS_REGISTRY = {
    "mse": MSELoss,
    "l1": L1Loss
  }

  def __init__(self, losses_config: List[Dict]):
    super().__init__()

    self.losses_list = []
    self.weights = []

    for loss_config in losses_config:
      loss_type = loss_config["type"]
      weight = loss_config.get("weight", 1.0)
      params = loss_config.get("params", {})

      if loss_type not in self.LOSS_REGISTRY:
        raise ValueError()

      loss_class = self.LOSS_REGISTRY[loss_type]
      loss_instance = loss_class(**params)

      self.losses_list.append(loss_instance)
      self.weights.append(weight)

      if isinstance(loss_instance, nn.Module):
        self.add_module(f"loss_{len(self.losses_list)-1}", loss_instance)

    print()
    for i, (loss, w) in enumerate(zip(self.losses_list, self.weights)):
      print(f" [{i}] {loss.name}: weight={w}")

  def forward(self, pred, target):
    """
    计算组合loss
    """
    total_loss = 0.0
    loss_dict = {}

    for loss, weight in zip(self.losses_list, self.weights):
      loss_value = loss(pred, target)
      if loss_value.dim() > 0:
        loss_value = loss_value.mean()
      total_loss = total_loss + weight * loss_value
      loss_dict[loss.name] = {
        "value": loss_value.item(),
        "weighted": weight * loss_value.item()
      }
    loss_dict["total"] = total_loss.item()
    return total_loss, loss_dict

  def get_config(self):
    config = {
      "loss_type": "combined",
      "losses": []
    }
    for loss, weight in zip(self.losses_list, self.weights):
      loss_config = {
        "type": loss.name,
        "weight": weight,
        "params": loss.get_coinfig()
      }
      config["losses"].append(loss_config)
    return config

# Loss Factory
class LossFactory:
  """
  从config创建Loss实例
  Config格式：
    # 单个loss
    {
      'loss_type': "variance_weighted",
      'min_weight': 0.1,
      'max_weight': 10.0
    }
    # 组合loss
    {
      'loss_type': "combined",
      'losses': [
        {"type": "variance_weighted", "weight": 1.0},
        {"type": "temporal_smooth", "weight": 0.1}
      ]
    }
  """
  @staticmethod
  def create_from_config(loss_config: Dict) -> BaseLoss:
    """
    从config字典创建loss实例
    Args:
      loss_config: loss配置字典

    Returns:
      loss: loss实例
    """
    loss_type = loss_config.get("loss_type", "mse")
    if loss_type == "combined":
      return CombinedLoss(loss_config["losses"])
    elif loss_type in CombinedLoss.LOSS_REGISTRY:
      loss_class = CombinedLoss.LOSS_REGISTRY[loss_type]
      params = {k: v for k, v in loss_config.items() if k != "loss_type"}
      return loss_class(**params)
    else:
      raise ValueError()
  
    @staticmethod
    def create_from_yaml(config_file: str) -> BaseLoss:
      """
      从yaml文件创建loss

      Args:
        config_file: yaml配置文件路径

      Returns:
        loss: loss实例
      """
      from omegaconf import OmegaConf
      loss_config = OmegaConf.load(config_file)
      return LossFactory.creat_from_config(loss_config)


    












