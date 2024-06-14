class myModel(x):
  @nn.compact
      def __call__(self, x):
  x = x.reshape((x.shape[0], -1))
  x = nn.Dense(features=10)(x)

while False:
  pass
