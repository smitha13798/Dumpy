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





class CNN(nn.Module):
    """A simple CNN model."""

    @nn.compact
    def __call__(self, x):
        x = nn.Conv(features=32, kernel_size=(3, 3))(x)
        x = nn.relu(x)
        for i in arr:
            x = nn.Conv(features=64, kernel_size=(3, 3))(x)
            x = nn.relu(x)
            x = nn.relu(x)
        x = x.reshape((x.shape[0], -1))  # flatten
        x = nn.Dense(features=256)(x)
        x = nn.relu(x)
        x = nn.Dense(features=64)(x)
        return x

class FunctionTypeEncoder(nn.Module):
    vocab_size: int
    embed_dim: int
    
    def setup(self):
        # Embedding layer for FunctionTypes
        self.embedding = nn.Embed(num_embeddings=self.vocab_size, features=self.embed_dim)
    
    def __call__(self, function_type_ids):
        return self.embedding(function_type_ids)
    

class FunctionTypeDecoder(nn.Module):
    vocab_size: int
    embed_dim: int
    
    def setup(self):
        # Simple dense layer to map embeddings to vocab size
        self.dense1 = nn.Dense(self.vocab_size)
    
    def __call__(self, embeddings):
        logits = self.dense1(embeddings)
        return logits


class FunctionTypeModel(nn.Module):
    vocab_size: int
    embed_dim: int

    def setup(self):
        self.encoder = FunctionTypeEncoder(self.vocab_size, self.embed_dim)
        self.decoder = FunctionTypeDecoder(self.vocab_size, self.embed_dim)

    def __call__(self, function_type_ids):
        embeddings = self.encoder(function_type_ids)
        logits = self.decoder(embeddings)
        return logits





class TrainState(train_state.TrainState):
    metrics: Metrics




class MLP(nn.Module):                    # create a Flax Module dataclass
  out_dims: int

  @nn.compact
  def __call__(self, x):
    x = x.reshape((x.shape[0], -1))
    x = nn.Dense(128)(y)                 # create inline Flax Module submodules
    x = nn.relu(x)
    x = nn.Dense(self.out_dims)(y)       # shape inference
    return x    




def test_rnn():
    model = SimpleRNN(input_size=28*28, hidden_size=128, num_layers=3, output_size=10)
    x = jax.random.normal(jax.random.PRNGKey(0), (64, 784, 1))
    vals = jax.random.normal(jax.random.PRNGKey(1), (64, 784, 783))
    x = jnp.concatenate([x, vals], axis=-1)
    variables = model.init(jax.random.PRNGKey(0), x)
    out = model.apply(variables, x)
    xshape = out.shape
    return x, xshape


testx, xdims = test_rnn()
print("Simple RNN size test: passed.")

class ExplicitMLP(nn.Module):
  features: Sequence[int]

  def setup(self):
    # we automatically know what to do with lists, dicts of submodules
    self.layers = [nn.Dense(feat) for feat in self.features]
    # for single submodules, we would just write:
    # self.layer1 = nn.Dense(feat1)

  def __call__(self, inputs):
    x = inputs
    for i, lyr in enumerate(self.layers):
      x = lyr(x)
      if i != len(self.layers) - 1:
        x = nn.relu(x)
    return x




