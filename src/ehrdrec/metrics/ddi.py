import numpy as np
import torch
from ehrdrec.metrics import Metric


class WeightedDDI(Metric):
    def __init__(self, ddi_weighted_matrix: np.ndarray):
        self.ddi_weighted_matrix = ddi_weighted_matrix
        self.value = 0.0
    
    def update(self, outputs: torch.Tensor, targets: torch.Tensor) -> None:
        if not outputs:
            raise ValueError("Outputs cannot be empty")
        # TODO: implement DDI calculation using the outputs and targets
    
    def compute(self):
        return self.value
    
    def reset(self):
        self.value = 0.0
