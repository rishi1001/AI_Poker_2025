import torch

def to_one_hot_vector(x: int, num_classes):
    return torch.eye(num_classes)[x]
