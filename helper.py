
def relErr(ytrue,ypred):
      return np.round(np.mean(np.mean(abs(ytrue-ypred)/ytrue)),3)
#     return accuracy_score(ytrue, ypred)/np.mean(ytrue)
get_slice = lambda i, size: range(i * size, (i + 1) * size)

def make_train_step(model, loss_fn, optimizer):
    # Builds function that performs a step in the train loop
    def train_step(x, y):
        # Sets model to TRAIN mode
        model.train()
        # Makes predictions
        yhat = model(x)
        # Computes loss
        loss = loss_fn(yhat.view(-1), y.view(-1))
        # Computes gradients
        loss.backward()
        # Updates parameters and zeroes gradients
        optimizer.step()
        optimizer.zero_grad()
        # Returns the loss
        return loss.item()
    # Returns the function that will be called inside the train loop
    return train_step

def val_estimator(model,val_loaer):
  with torch.no_grad():
    model.eval()
    val_losses_L1 = []
    for x_val, y_val in val_loader:
        model.eval()
      # x_val, y_val = next(iter(val_loader))
        x_val = x_val.to(device)
        y_val = y_val.to(device)
        y_hat = model(x_val)
        val_loss = loss_measure(y_val,y_hat)   #L1
        val_losses_L1.append(val_loss.item()) 
    print(st.mean(val_losses_L1))
    return val_losses_L1



