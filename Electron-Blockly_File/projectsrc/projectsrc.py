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


#Model+
class myModel(x):
  @nn.compact
      def __call__(self, x):
  x = x.reshape((x.shape[0], -1))
  x = nn.Dense(features=10)(x)

while False:
  pass
#Model-class Metrics(metrics.Collection):
    accuracy: metrics.Accuracy
    loss: metrics.Average.from_output('loss')


class TrainStates(train_state.TrainState):
    metrics: Metrics


def create_train_state(module, rng, learning_rate, momentum):
    """Creates an initial `TrainState`."""
    params = module.init(rng, jnp.ones([1, 28, 28, 1]))['params']  # initialize parameters by passing a template image
    tx = optax.sgd(learning_rate, momentum)
    return TrainStates.create(
        apply_fn=module.apply, params=params, tx=tx,
        metrics=Metrics.empty())


model = CNN()
learning_rate = 0.01
momentum = 0.9
init_rng = jax.random.key(0)
state = create_train_state(model, init_rng, learning_rate, momentum)
