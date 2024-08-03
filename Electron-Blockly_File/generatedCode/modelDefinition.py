class CNN(nn.Module):
  @nn.compact
  def __call__(self,x):
      x = nn.Conv(features=32, kernel_size=(3, 3))(x)
      x = nn.relu(x)
      x = nn.avg_pool(window_shape=(2, 2), strides=(2, 2))(x)
      x = x.reshape((x.shape[0], -1))
      x = x.reshape((x.shape[0], -1))
      x = nn.Dense()(x)
