import torch
from torch.utils.data import DataLoader


def evaluate_accuracy(model: torch.nn.Module, loader: DataLoader, device=None):
    """Compute accuracy of input model over all samples from the loader.
    
    Args:
        model (torch.nn.Module): NN model
        loader (DataLoader): Data loader to evaluate on
        device (torch.device), optional:
            Device to use, by default None.
            If None uses cuda if available else cpu.
    Returns:
        float: Accuracy in [0,1]
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model.eval()

    y_preds = []
    y_trues = []

    for data in loader:  # Iterate in batches over the training/test dataset.
        data.to(device)
        out = model(data.x, data.edge_index, data.batch)
        y_preds.append(out.argmax(dim=1))  # Use the class with highest probability.
        y_trues.append(data.y)  # Check against ground-truth labels.

    y_pred = torch.cat(y_preds).flatten()
    y_true = torch.cat(y_trues).flatten()

    return torch.sum(y_pred == y_true).item() / len(y_true)
