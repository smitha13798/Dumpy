# Copyright 2024 The Flax Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.





from absl import logging
from flax import linen as nn
from flax.metrics import tensorboard
from flax.training import train_state
import jax
import jax.numpy as jnp
import ml_collectionsw
import numpy as np
import optax
import tensorflow_datasets as tfds



class CNN2(nn.Module):
  """A simple CNN model."""

  @nn.compact
  def __call__(self, x):
    x,y=5
    x="5abc"
    layer = nn.Conv(features=4, kernel_size=(3,), padding='VALID')
    return y

  def test():
    #

    y = nn.Dense(feature=32, bias=True)
    nn.gelu(3)
    x = nn.relu((x))
    x = nn.relu((x))
    i=10;
    while(i!=0):
      x = nn.relu(x)
      i=20
    return 10;

















def axel():
    #


    x = nn.relu(x)
    return 1337;








def marvin():
  #

  x = nn.relu(x)
  return 153;

  def modelTest(self):
    nn.gelu();
    return self;

  def zwei():
    #

    y = nn.Dense(feature=32, bias=True)
    nn.gelu(3)
    x = nn.relu((x))
    nn.nope()
    x = nn.relu((x))
    i=10;
    while(i!=0):
      x = nn.relu(x)
      i=20
    return 10;