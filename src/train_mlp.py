import pytorch_lightning as pl
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset, random_split
import pandas as pd
# Example synthetic regression dataset
def get_data(n_samples=5000):
    X = torch.randn(n_samples, 10)
    y = X.sum(dim=1, keepdim=True) + 0.1 * torch.randn(n_samples, 1)
    return X, y

class MLPRegressor(pl.LightningModule):
    def __init__(self, input_dim=10, hidden_dim=64, lr=1e-3):
        super().__init__()
        self.save_hyperparameters()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        self.loss_fn = nn.MSELoss()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)
        self.log('val_loss', loss)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)

if __name__ == "__main__":
    X, y = get_data()
    dataset = TensorDataset(X, y)
    train_len = int(0.8 * len(dataset))
    val_len = len(dataset) - train_len
    train_ds, val_ds = random_split(dataset, [train_len, val_len])
    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=32)

    train_X = train_ds.dataset.tensors[0][train_ds.indices]
    train_y = train_ds.dataset.tensors[1][train_ds.indices]
    train_df = pd.DataFrame(train_X.numpy(), columns=[f"feature_{i}" for i in range(train_X.shape[1])])
    train_df["target"] = train_y.numpy().flatten()
    train_df.to_csv("train_ds.csv", index=False)

    model = MLPRegressor(input_dim=10, hidden_dim=16, lr=1e-3)
    trainer = pl.Trainer(max_epochs=1000, accelerator="auto", log_every_n_steps=10)
    trainer.fit(model, train_loader, val_loader)
    torch.save(model.state_dict(), "mlp_model.pt")