function generate_randomePRNG(seed) {
       key = jax.random.PRNGKey(seed)
       return key
}
