class CNN(nn.Module):
  @nn.compact
  def __call__(self,x):
      x = nn.tanh(x)
      x = nn.relu(x)
      x = nn.BatchNorm()(x)
