# Flow LVM

A new type of latent variable model using diffusion / flow sampling as encoder to generate the latent distribution p(z|x). This allows for much more flexible posterios, than restricting to Gaussians.

Vanilla refers to toy setup on toy tasks. The Meta* parts apply this concept to a Bayesian meta-learning settings.

### Relevant Papers

- [NETS: A Non-Equilibrium Transport Sampler](https://arxiv.org/abs/2410.02711)
- [Variational Auto-Encoder](https://arxiv.org/abs/1906.02691)

### Vanilla Setup

![alt text](assets/vanilla.gif)

