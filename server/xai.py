import torch

def compute_saliency(model, input_tensor):
    """
    Compute saliency map for ECG input.

    Args:
        model: trained model
        input_tensor: shape (1, 1, 1800)

    Returns:
        saliency map (1D array)
    """
    model.eval()

    input_tensor = input_tensor.clone().detach()
    input_tensor.requires_grad = True

    output = model(input_tensor)
    pred_class = output.argmax(dim=1)

    # Get score for predicted class
    score = output[0, pred_class.item()]

    # Backprop
    model.zero_grad()
    score.backward()

    # Gradient wrt input
    saliency = input_tensor.grad.data.abs().squeeze().cpu().numpy()

    return saliency