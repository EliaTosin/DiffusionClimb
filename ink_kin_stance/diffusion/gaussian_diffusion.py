import torch
import torch.nn.functional as F


def get_beta_schedule(num_timesteps, beta_start=1e-4, beta_end=0.02):
    """Linear beta schedule."""
    return torch.linspace(beta_start, beta_end, num_timesteps)


def extract(a, t, x_shape):
    """Extract values from *a* at indices *t*, reshaped for broadcasting."""
    batch_size = t.shape[0]
    out = a.gather(-1, t)
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))


class GaussianDiffusion:
    def __init__(self, num_timesteps=500, beta_start=1e-4, beta_end=0.02,
                 device="cuda"):
        self.num_timesteps = num_timesteps
        self.device = device

        self.betas = get_beta_schedule(num_timesteps, beta_start, beta_end).to(device)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)

        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)

        self.posterior_variance = (
            self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)

    # ------------------------------------------------------------------
    # Forward process
    # ------------------------------------------------------------------

    def q_sample(self, x_0, t, noise=None):
        """Forward diffusion: q(x_t | x_0)."""
        if noise is None:
            noise = torch.randn_like(x_0)
        sqrt_ac = extract(self.sqrt_alphas_cumprod, t, x_0.shape)
        sqrt_omac = extract(self.sqrt_one_minus_alphas_cumprod, t, x_0.shape)
        return sqrt_ac * x_0 + sqrt_omac * noise

    def p_losses(self, model, x_0, t, condition, noise=None):
        """Training loss: predict noise."""
        if noise is None:
            noise = torch.randn_like(x_0)
        x_t = self.q_sample(x_0, t, noise)
        predicted_noise = model(x_t, t, condition)
        return F.mse_loss(predicted_noise, noise)

    # ------------------------------------------------------------------
    # Reverse process
    # ------------------------------------------------------------------

    @torch.no_grad()
    def p_sample(self, model, x_t, t, condition):
        """Reverse diffusion: sample x_{t-1} from x_t."""
        betas_t = extract(self.betas, t, x_t.shape)
        sqrt_omac_t = extract(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape)
        sqrt_recip_t = extract(self.sqrt_recip_alphas, t, x_t.shape)

        predicted_noise = model(x_t, t, condition)
        model_mean = sqrt_recip_t * (x_t - betas_t * predicted_noise / sqrt_omac_t)

        if t[0] == 0:
            return model_mean
        posterior_var_t = extract(self.posterior_variance, t, x_t.shape)
        noise = torch.randn_like(x_t)
        return model_mean + torch.sqrt(posterior_var_t) * noise

    @torch.no_grad()
    def sample(self, model, condition, shape):
        """Generate samples from noise (full DDPM, all timesteps)."""
        device = condition.device
        x = torch.randn(shape, device=device)
        for t in reversed(range(self.num_timesteps)):
            t_batch = torch.full((shape[0],), t, device=device, dtype=torch.long)
            x = self.p_sample(model, x, t_batch, condition)
        return x

    @torch.no_grad()
    def ddim_sample(self, model, condition, shape, ddim_steps=50, eta=0.0):
        """Fast DDIM sampling with fewer steps."""
        device = condition.device
        x = torch.randn(shape, device=device)

        step_size = self.num_timesteps // ddim_steps
        timesteps = list(reversed(list(range(0, self.num_timesteps, step_size))))

        for i, t in enumerate(timesteps):
            t_batch = torch.full((shape[0],), t, device=device, dtype=torch.long)
            predicted_noise = model(x, t_batch, condition)

            alpha_t = self.alphas_cumprod[t]
            alpha_prev = (
                self.alphas_cumprod[timesteps[i + 1]]
                if i + 1 < len(timesteps)
                else torch.tensor(1.0, device=device)
            )

            x0_pred = (x - torch.sqrt(1 - alpha_t) * predicted_noise) / torch.sqrt(alpha_t)

            sigma = eta * torch.sqrt(
                (1 - alpha_prev) / (1 - alpha_t) * (1 - alpha_t / alpha_prev)
            )
            dir_xt = torch.sqrt(
                torch.clamp(1 - alpha_prev - sigma ** 2, min=0.0)
            ) * predicted_noise

            x = torch.sqrt(alpha_prev) * x0_pred + dir_xt
            if sigma > 0 and i + 1 < len(timesteps):
                x = x + sigma * torch.randn_like(x)

        return x


class DeterministicGaussianDiffusion(GaussianDiffusion):
    def __init__(self, num_timesteps=500, beta_start=1e-4, beta_end=0.02, device="cuda"):
        super().__init__(num_timesteps, beta_start, beta_end, device)
        self.gen = torch.Generator(device=device)
        self.gen.manual_seed(42)

    # ------------------------------------------------------------------
    # Reverse process
    # ------------------------------------------------------------------

    @torch.no_grad()
    def p_sample(self, model, x_t, t, condition):
        """Reverse diffusion: sample x_{t-1} from x_t."""
        betas_t = extract(self.betas, t, x_t.shape)
        sqrt_omac_t = extract(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape)
        sqrt_recip_t = extract(self.sqrt_recip_alphas, t, x_t.shape)

        predicted_noise = model(x_t, t, condition)
        model_mean = sqrt_recip_t * (x_t - betas_t * predicted_noise / sqrt_omac_t)

        if t[0] == 0:
            return model_mean
        posterior_var_t = extract(self.posterior_variance, t, x_t.shape)
        noise = torch.randn(x_t.shape, dtype=x_t.dtype, device=x_t.device, generator=self.gen)

        return model_mean + torch.sqrt(posterior_var_t) * noise

    @torch.no_grad()
    def sample(self, model, condition, shape):
        """Generate samples from noise (full DDPM, all timesteps)."""
        device = condition.device
        x = torch.randn(shape, device=device, generator=self.gen)
        for t in reversed(range(self.num_timesteps)):
            t_batch = torch.full((shape[0],), t, device=device, dtype=torch.long)
            x = self.p_sample(model, x, t_batch, condition)
        return x
