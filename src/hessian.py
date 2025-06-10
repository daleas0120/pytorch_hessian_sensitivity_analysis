import torch

def compute_hessian(model, inputs, loss_fn):
    """
    Computes the Hessian matrix of the loss with respect to the model parameters.

    Args:
        model: torch.nn.Module
        inputs: tuple (input, target) for the model and loss function
        loss_fn: loss function (e.g., torch.nn.CrossEntropyLoss())

    Returns:
        hessian: torch.Tensor of shape (num_params, num_params)
    """
    model.zero_grad()
    input_data, target = inputs
    output = model(input_data)
    loss = loss_fn(output, target)
    grads = torch.autograd.grad(loss, model.parameters(), create_graph=True)
    grads_flat = torch.cat([g.contiguous().view(-1) for g in grads])
    num_params = grads_flat.numel()
    hessian = torch.zeros(num_params, num_params, device=grads_flat.device)

    for idx in range(num_params):
        grad2 = torch.autograd.grad(grads_flat[idx], model.parameters(), retain_graph=True)
        grad2_flat = torch.cat([g.contiguous().view(-1) for g in grad2])
        hessian[idx] = grad2_flat

    return hessian