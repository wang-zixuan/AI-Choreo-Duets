import torch
from model.transformer import DanceTransformer



class Pipeline:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.network = DanceTransformer()
        self.loss = 0

        self.init_training_settings()

    def init_training_settings(self):
        self.network.train()
        self.cri_regression_loss = torch.nn.MSELoss()
        self.setup_optimizers()
        self.setup_schedulers()

    def setup_optimizers(self):
        optim_params = []
        for k, v in self.network.named_parameters():
            if v.requires_grad:
                optim_params.append(v)

        self.optimizer = torch.optim.Adam(optim_params)

    def setup_schedulers(self):
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[10, 20])

    def feed_data(self, data):
        self.dancer1_data = data['dancer1'].to(self.device)
        self.dancer2_data = data['dancer2'].to(self.device)

    def optimize_parameters(self):
        self.network.zero_grad()
        pred_data1, pred_data2 = self.network(self.dancer1_data, self.dancer2_data)

        self.loss = self.cri_regression_loss(self.dancer1_data, pred_data1) \
                + self.cri_regression_loss(self.dancer2_data, pred_data2)

        self.loss.backward()
        self.optimizer.step()

    def update_learning_rate(self):
        self.scheduler.step()
