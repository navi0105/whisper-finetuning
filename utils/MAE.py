from tqdm import tqdm

def MAE(pred: list, ref: list) -> float:
    n = len(ref)
    error_sum = 0.0
    for x, y in zip(pred, ref):
        error_sum += abs(y - x)
    
    return error_sum / n