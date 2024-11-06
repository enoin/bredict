def z_norm(tensor):
    mean, std = tensor.mean(dim=-1, keepdim=True), tensor.std(dim=-1, keepdim=True)
    normed_seqs = (tensor - mean) / std
    return normed_seqs


def normalize_range(tensor, a=0, b=1):
    min_val = tensor.min()
    max_val = tensor.max()
    normalized_tensor = a + (tensor - min_val) * (b - a) / (max_val - min_val)
    return normalized_tensor


def min_max_scale(tensor):
    min_val = tensor.min()
    max_val = tensor.max()
    return (tensor - min_val) / (max_val - min_val)

