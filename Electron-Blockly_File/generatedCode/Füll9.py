def train_step(model, data, loss_fn, optimizer):
    # Forward pass
    predictions = model(data)
    loss = loss_fn(predictions, data['labels'])
    # Backward pass and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
