from typing import Callable, Iterable, Tuple

import torch
from torch.optim import Optimizer


class AdamW(Optimizer):
    """
    Implements AdamW optimizer with weight decay fix as introduced in 
    'Decoupled Weight Decay Regularization' paper: https://arxiv.org/abs/1711.05101
    
    The main differences from Adam:
    * Weight decay is decoupled from the optimization steps
    * The order of operations is slightly different to maintain consistency
    """
    def __init__(
            self,
            params: Iterable[torch.nn.parameter.Parameter],
            lr: float = 1e-3,
            betas: Tuple[float, float] = (0.9, 0.999),  # exponential decay rates for moment estimates
            eps: float = 1e-6,  # term added for numerical stability
            weight_decay: float = 0.0,  # decoupled weight decay coefficient
            correct_bias: bool = True,  # whether to correct bias in Adam (paper suggests True)
    ):
        # Validate parameters
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {} - should be >= 0.0".format(lr))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter: {} - should be in [0.0, 1.0[".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter: {} - should be in [0.0, 1.0[".format(betas[1]))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {} - should be >= 0.0".format(eps))
            
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, correct_bias=correct_bias)
        super().__init__(params, defaults)

    def step(self, closure: Callable = None):
        """
        Performs a single optimization step.
        
        Args:
            closure (Callable, optional): A closure that reevaluates the model and returns the loss.
        
        The optimization algorithm follows these steps:
        1. Compute gradients (done externally before calling step)
        2. Update biased first moment estimate (momentum)
        3. Update biased second raw moment estimate
        4. Apply bias correction to moment estimates
        5. Update parameters with weight decay done separately
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError("Adam does not support sparse gradients, please consider SparseAdam instead")

                # Get or initialize optimizer state for this parameter
                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    # Initialize first moment (momentum) - running average of gradients
                    state["exp_avg"] = torch.zeros_like(p.data)
                    # Initialize second moment - running average of squared gradients
                    state["exp_avg_sq"] = torch.zeros_like(p.data)

                # Get optimizer parameters for this group
                beta1, beta2 = group["betas"]  # Exponential decay rates
                eps = group["eps"]  # For numerical stability
                lr = group["lr"]  # Learning rate
                weight_decay = group["weight_decay"]  # Decoupled weight decay
                correct_bias = group["correct_bias"]  # Whether to correct bias in Adam

                state["step"] += 1

                # Update first and second moments of the gradients
                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                # Update biased first moment estimate (momentum)
                # m_t = beta1 * m_{t-1} + (1 - beta1) * g_t
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                # Update biased second raw moment estimate
                # v_t = beta2 * v_{t-1} + (1 - beta2) * g_t^2
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                # Apply bias correction to moment estimates
                # Corrects for initialization bias in first and second moments
                if correct_bias:
                    # Bias correction for moment estimates
                    # m_t_hat = m_t / (1 - beta1^t)
                    # v_t_hat = v_t / (1 - beta2^t)
                    bias_correction1 = 1 - beta1 ** state["step"]
                    bias_correction2 = 1 - beta2 ** state["step"]
                    step_size = lr * (bias_correction2 ** 0.5) / bias_correction1
                else:
                    step_size = lr

                # Update parameters using Adam update rule
                # theta_t = theta_{t-1} - lr * m_t_hat / (sqrt(v_t_hat) + eps)
                denom = exp_avg_sq.sqrt().add_(eps)
                p.data.addcdiv_(exp_avg, denom, value=-step_size)

                # Add decoupled weight decay after the main gradient-based updates
                # This is the key difference between Adam and AdamW
                # Instead of applying L2 regularization through the gradients,
                # we explicitly perform weight decay after the Adam update
                if weight_decay != 0:
                    p.data.add_(p.data, alpha=-lr * weight_decay)

        return loss
