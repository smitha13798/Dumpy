from flax import linen as nn
class CNN(nn.Module):
  """A simple CNN model."""

  @nn.compactx = None


x = None

# Describe this function...
def __call__(x):
  x = x.reshape((x.shape[0], -1))
  x = nn.Dense(features=10)(x)
  x = nn.MaxPool(window_shape=(2, 2), strides=(2, 2))(x)
  return x


class MyClass:
