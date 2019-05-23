import torch
from torch.optim import Optimizer 


class GeoSGD(Optimizer):
    r"""Implements geodesic gradient descent on the Stiefel manifold of
    orthogonal matrices.
    
    `On the importance of initialization and momentum in deep learning`__.
    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float): learning rate
    Example:
        >>> optimizer = torch.optim.GeoSGD(model.parameters(), lr=0.1)
        >>> optimizer.zero_grad()
        >>> loss_fn(model(input), target).backward()
        >>> optimizer.step()
    __ https://arxiv.org/abs/1702.00071
    __ https://arxiv.org/abs/1611.00035
    """

    def __init__(self, params, lr=1e-5):
        defaults = dict(lr=lr)
        super(GeoSGD, self).__init__(params, defaults)

    def step(self, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                P = p.data
                G = p.grad.data
                A = torch.mm(G.t(), P) - torch.mm(P.t(), G)
                I = torch.eye(A.shape[0])
                if A.is_cuda:
                    I = I.cuda()
                lr = group['lr']
                cayley = torch.mm(torch.inverse(I+(lr/2.)*A), I-(lr/2.)*A)
                p.data.copy_(torch.mm(cayley, P))

        return loss

