
def koray():
    x = np_solver(10)

def test():
    y =6
    x = nn.Dense(feature=32, bias=True)
    nn.gelu(3)
    x = nn.relu(x)
    x = nn.relu(x)
    return 5




class CNN(nn.Module):
  """A simple CNN model."""

  @nn.compact
  def __call__(self, x):
    y= 5
    x = nn.Conv(features=32, kernel_size=(3, 3))(x)
    x = nn.relu(x)
    x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
    x = nn.Conv(features=64, kernel_size=(3, 3))(x)
    x = nn.relu(x)
    x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
    x = x.reshape((x.shape[0], -1))  # flatten
    x = nn.relu(x)
    x = nn.relu(y)


    return y

  def test(a, b):
      y = 10
      x = nn.Dense(feature=32, bias=True)
      nn.gelu(3)
      x = nn.relu(x)
      x = nn.relu(x)

      return 5

