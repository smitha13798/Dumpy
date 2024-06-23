import jax
from clu import metrics
from flax.training import train_state  # Useful dataclass to keep train state
from flax import struct  # Flax dataclasses
import optax
import flax.linen as nn
import jax.numpy as jnp


class CNN(nn.Module):
    @nn.compact
    def __call__(self, x):
        x = nn.Conv(features=32, kernel_size=(3, 3))(x)
        x = nn.relu(x)
        x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
        x = nn.Conv(features=64, kernel_size=(3, 3))(x)
        x = nn.relu(x)
        x = nn.avg_pool(x, window_shape=(2, 2), strides=(2, 2))
        x = x.reshape((x.shape[0], -1))  # flatten
        x = nn.Dense(features=256)(x)
        x = nn.relu(x)
        x = nn.Dense(features=10)(x)
        x = nn.log_softmax(x)
        return x


#modelDefinition+
class CNN(nn.Module):
  @nn.compact
  def __call__(self,x):
      x = nn.tanh(x)
      x = nn.relu(x)
      x = nn.BatchNorm()(x)
#modelDefinition-


#dataloaderDefinition+

#dataloaderDefinition-





