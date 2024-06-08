class myModule:
  x = x.reshape((x.shape[0], -1))
  x = nn.Dense(features=10)(x)
  x = nn.MaxPool(window_shape=(2, 2), strides=(2, 2))(x)

class Paloma:
  while False:
    pass
