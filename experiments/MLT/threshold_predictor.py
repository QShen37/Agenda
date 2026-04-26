import torch
from .model import DualHeadThresholdNet
from .embedding import embed_task

class ThresholdPredictor:
    def __init__(self, model_path):
        self.model = DualHeadThresholdNet(
            input_dim=384,
            hidden_dim=128,
            N = 7
        )
        self.model.load_state_dict(torch.load(model_path, map_location="cpu"))
        self.model.eval()

    def predict(self, task):
        e = embed_task(task)
        e = torch.tensor(e, dtype=torch.float32).unsqueeze(0)

        with torch.no_grad():
            beta, gamma = self.model(e)

        return beta.item(), gamma.item()
