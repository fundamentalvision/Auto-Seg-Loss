import torch
import numpy as np

class TruncatedNormal(object):
    '''
        A Class of the Truncated Normal Distribution.
        mu: average value of the original gaussian distribution
        sigma: standard derivation of the original gaussian distribution
        a: min value
        b: max value
    '''
    def __init__(self, mu, sigma, a, b):
        super(TruncatedNormal, self).__init__()
        # mu, sigma : (n_param,)
        # a, b : float
        self.mu = mu
        self.sigma = sigma
        self.a = a
        self.b = b
        
        self.n_params = len(mu)
        
        assert self.mu.dim() == 1
        assert b > a
        
        self.normal_sampler = torch.distributions.normal.Normal(torch.zeros(self.n_params, device=torch.cuda.current_device()), torch.ones(self.n_params, device=torch.cuda.current_device()))
        self.uniform_sampler = torch.distributions.uniform.Uniform(torch.zeros(self.n_params, device=torch.cuda.current_device()), torch.ones(self.n_params, device=torch.cuda.current_device()))
        
        
    def sample(self, batch_size):
        alpha = (-self.mu + self.a) / self.sigma
        beta = (-self.mu + self.b) / self.sigma
        
        uniform = self.uniform_sampler.sample([batch_size])
        
        alpha_normal_cdf = self.normal_sampler.cdf(alpha)
        p = (self.normal_sampler.cdf(beta) - alpha_normal_cdf).view(1, -1) * uniform + alpha_normal_cdf.view(1, -1)
        
        epsilon = torch.finfo(p.dtype).eps
        v = torch.clamp(2 * p - 1, -1 + epsilon, 1 - epsilon)
        
        x = self.mu.view(1, -1) + (2**0.5) * torch.erfinv(v) * self.sigma.view(1, -1)
        x = torch.clamp(x, self.a, self.b)
        
        return x.detach()
    
    def log_prob(self, x):
        alpha = (-self.mu + self.a) / self.sigma
        beta = (-self.mu + self.b) / self.sigma
        
        normal_x = (x - self.mu) / self.sigma
        
        down = (self.normal_sampler.cdf(beta) - self.normal_sampler.cdf(alpha)).log().view(1, -1)
        
        up = (1.0 / self.sigma).log() + self.normal_sampler.log_prob(normal_x)
        
        l_prob = up - down
        
        l_prob = l_prob.where((x >= self.a) & (x <= self.b), torch.tensor(-np.inf, device=torch.cuda.current_device()))
        
        return l_prob
