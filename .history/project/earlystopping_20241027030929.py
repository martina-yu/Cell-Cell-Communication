class EarlyStopping:
    def __init__(self, patience=10, min_delta=0):
        """
        Args:
            patience (int): How many epochs to wait after last time the validation loss improved.
            min_delta (float): Minimum change in the monitored quantity to qualify as an improvement.
        """
        self.patience = patience
        self.min_delta = min_delta
        self.best_val_loss = None
        self.counter = 0
        self.best_epoch = 0

    def check(self, val_loss, current_epoch):
        """
        Checks if the validation loss has improved and whether training should be stopped early.
        Args:
            val_loss (float): The current epoch's validation loss.
            current_epoch (int): The current epoch number.
        Returns:
            bool: True if training should stop, False otherwise.
        """
        if self.best_val_loss is None:
            self.best_val_loss = val_loss
            self.best_epoch = current_epoch
        elif val_loss < self.best_val_loss - self.min_delta:
            self.best_val_loss = val_loss
            self.counter = 0
            self.best_epoch = current_epoch
        else:
            self.counter += 1
            if self.counter >= self.patience:
                return True  # Stop training
        return False  # Continue training

    def get_best_epoch(self):
        return self.best_epoch


def test_with_early_stopping(data, device, max_epochs=500, patience=10):
    best_model = RegularGAT(
        data.x.shape[1],
        data.labels.shape[1],
        data.y.shape[1],
        best_params['hidden_channels'],
        best_params['heads'],
        best_params['dropout'],
    ).to(device)

    optimizer = torch.optim.Adam(best_model.parameters(), lr=best_params["lr"])

    early_stopping = EarlyStopping(patience=patience, min_delta=1e-4)  # Adjust `min_delta` as needed

    best_epoch = 0
    val_losses = []
    train_losses = []

    for epoch in range(1, max_epochs + 1):
        # Training step
        train_total_loss, train_attention_weights, train_attention_scores = train(data, device)
        train_losses.append(train_total_loss)

        # Validation step
        val_total_loss, val_attention_weights, val_attention_scores = validate(data, device)
        val_losses.append(val_total_loss)

        print(f"Epoch {epoch}, Validation Loss: {val_total_loss:.4f}")

        # Early stopping check
        if early_stopping.check(val_total_loss, epoch):
            print(f"Stopping early at epoch {epoch}")
            break

        if epoch == 1 or epoch % 100 == 0:
            torch.save(best_model.state_dict(), f'{checkpoint}/best_model_epoch_{epoch + 1}_{select_family}_{brain_data}.pth')

    # Get the best epoch where the validation loss was the best
    best_epoch = early_stopping.get_best_epoch()
    print(f"Best epoch selected: {best_epoch}")

    # Save the best model parameters and return the best epoch
    torch.save(best_model.state_dict(), f'{checkpoint}/best_model_epoch_{best_epoch}_{select_family}_{brain_data}.pth')

    return best_epoch, val_losses, train_losses