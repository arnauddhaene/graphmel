import torch


def standardized(t: torch.tensor, mean: torch.tensor, std: torch.tensor) -> torch.tensor:
    """
    Standardize tensor following given mean and standard deviation

    Args:
        t (torch.tensor): tensor to be standardized
        mean (torch.tensor): mean
        std (torch.tensor): standard deviation

    Returns:
        torch.tensor: standardized tensor
    """
    
    return t.sub_(mean).div_(std)


def extract_study_phase(n: str) -> int:
    # split study name into 'pre'/'post' and it's associated number
    status, number = n.split('-')
    # format it with 0 being the treatment start
    return (-1 if status == 'pre' else 1) * int(number)


def format_number_header(heading: str, spotlight: str, footer: str) -> str:
    return f"""
        <div class="container-number">
            <div class="number-header"> {heading} </div>
            <h1 class="number"> {spotlight} </h1>
            <div class="number-footer"> {footer} </div>
        </div>
    """
