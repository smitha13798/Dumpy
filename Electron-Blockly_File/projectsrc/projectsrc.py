

#modelDefinition+
function generate_randomePRNG(seed) {
       key = jax.random.PRNGKey(seed)
       return key
}


generate_randomePRNG('');

x = x.reshape((x.shape[0], -1))
#modelDefinition-
#modelDefinition-
#modelDefinition-
#modelDefinition-


#dataloaderDefinition+
function generate_randomePRNG(seed) {
       key = jax.random.PRNGKey(seed)
       return key
}


generate_randomePRNG('');

data_loader = DataLoader(dataset, batch_size=32, shuffle=True)
#dataloaderDefinition-
#dataloaderDefinition-
















