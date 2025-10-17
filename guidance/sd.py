from diffusers import DDIMScheduler, DDIMInverseScheduler, StableDiffusionPipeline
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random

from peft import LoraConfig


class StableDiffusion(nn.Module):
    def __init__(self, args, t_range=[0.02, 0.98]):
        super().__init__()

        self.device = args.device
        self.dtype = args.precision
        print(f'[INFO] Loading Stable Diffusion...')

        model_key = "stabilityai/stable-diffusion-2-1-base"
        pipe = StableDiffusionPipeline.from_pretrained(
            model_key, torch_dtype=self.dtype,
        )

        pipe.to(self.device)
        self.vae = pipe.vae.eval()
        self.tokenizer = pipe.tokenizer
        self.text_encoder = pipe.text_encoder
        self.unet = pipe.unet.eval()
        
        # Freeze models
        for p in self.vae.parameters():
            p.requires_grad_(False)
        for p in self.unet.parameters():
            p.requires_grad_(False)
        
        # Schedulers
        self.scheduler = DDIMScheduler.from_pretrained(
            model_key, subfolder="scheduler", torch_dtype=self.dtype,
        )
        self.scheduler.alphas_cumprod = self.scheduler.alphas_cumprod.to(self.device)
        
        self.inverse_scheduler = DDIMInverseScheduler.from_pretrained(
            model_key, subfolder="scheduler", torch_dtype=self.dtype,
        )
        self.inverse_scheduler.alphas_cumprod = self.inverse_scheduler.alphas_cumprod.to(self.device)

        del pipe

        self.num_train_timesteps = self.scheduler.config.num_train_timesteps
        self.min_step = int(self.num_train_timesteps * t_range[0])
        self.max_step = int(self.num_train_timesteps * t_range[1])
        self.alphas = self.scheduler.alphas_cumprod
        
        print(f'[INFO] Loaded Stable Diffusion!')
        
        # Initialize VSD components if needed
        if args.loss_type == "vsd":
            self._init_vsd_components(args.lora_rank)
    
    def _init_vsd_components(self, lora_rank=4):
        """Initialize LoRA for VSD"""
        print(f"[INFO] Initializing VSD with LoRA rank={lora_rank}")
        
        self.unet.requires_grad_(False)
        
        unet_lora_config = LoraConfig(
            r=lora_rank,
            lora_alpha=lora_rank,
            init_lora_weights="gaussian",
            target_modules=["to_k", "to_q", "to_v", "to_out.0"],
        )
        
        self.unet.add_adapter(unet_lora_config)
        self.lora_layers = list(filter(lambda p: p.requires_grad, self.unet.parameters()))

    @torch.no_grad()
    def get_text_embeds(self, prompt):
        """Get text embeddings from prompt"""
        inputs = self.tokenizer(
            prompt, 
            padding='max_length', 
            max_length=self.tokenizer.model_max_length, 
            return_tensors='pt'
        )
        embeddings = self.text_encoder(inputs.input_ids.to(self.device))[0]
        return embeddings
    
    def get_noise_preds(self, latents_noisy, t, text_embeddings, guidance_scale=7.5):
        """
        Predict noise with Classifier-Free Guidance (CFG)
        
        CFG formula: noise_pred = uncond + guidance_scale * (cond - uncond)
        """
        latent_model_input = torch.cat([latents_noisy] * 2)
        tt = torch.cat([t] * 2)
        
        noise_pred = self.unet(latent_model_input, tt, encoder_hidden_states=text_embeddings).sample
        noise_pred_uncond, noise_pred_pos = noise_pred.chunk(2)
        
        # Classifier-Free Guidance
        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_pos - noise_pred_uncond)
        
        return noise_pred
    
    def get_sds_loss(self, latents, text_embeddings, guidance_scale=7.5):
        """
        Score Distillation Sampling (SDS) Loss
        
        Reference: DreamFusion (https://arxiv.org/abs/2209.14988)
        """
        B = latents.shape[0]
        t = torch.randint(self.min_step, self.max_step + 1, (B,), dtype=torch.long, device=self.device)
        noise = torch.randn_like(latents)
        latents_noisy = self.scheduler.add_noise(latents, noise, t)
        
        noise_pred = self.get_noise_preds(latents_noisy, t, text_embeddings, guidance_scale)
        
        # Compute SDS gradient: (ε_θ(x_t, c, t) - ε)
        # The gradient cannot be directly assigned to latents.grad, so we need to convert it back to a loss
        # We want a loss function L that satisfying ∂L/∂x_0 = ε_θ - ε
        # So assume L = 0.5 * ||x_0 - target||^2
        # -> x_0 - target = ε_θ - ε
        # -> target = (x_0 - (ε_θ - ε))
        grad = noise_pred - noise

        # Detach target to cut the computational graph. Without detach, 
        # loss would depend on latents through two paths:
        # 1. Direct: loss ← (latents - target)
        # 2. Indirect: loss ← target ← grad ← noise_pred ← U-Net ← latents_noisy ← latents
        # This would give ∂loss/∂latents = grad * ∂grad/∂latents instead of just grad
        target = (latents - grad).detach()
        
        # Compute loss: this will give gradient = latents - target = latents - (latents - grad) = grad
        loss = 0.5 * F.mse_loss(latents, target, reduction='mean')
        
        return loss
    
    def get_vsd_loss(self, latents, text_embeddings, guidance_scale=7.5, lora_loss_weight=1.0):
        """
        Variational Score Distillation (VSD) Loss
        
        Reference: ProlificDreamer (https://arxiv.org/abs/2305.16213)
        """
        # One-time initialization of LoRA-B weights to avoid zero output
        if not hasattr(self, '_vsd_lora_initialized'):
            print("[VSD] 初始化 LoRA-B 權重...")
            initialized_count = 0
            for name, module in self.unet.named_modules():
                if 'lora_B' in name and hasattr(module, 'default'):
                    if hasattr(module.default, 'weight'):
                        nn.init.normal_(module.default.weight, mean=0.0, std=0.01)
                        initialized_count += 1
            print(f"[VSD] 已初始化 {initialized_count} 個 LoRA-B 模組")
            self._vsd_lora_initialized = True
        
        B = latents.shape[0]
        
        # Sample random timestep
        t = torch.randint(self.min_step, self.max_step + 1, (B,), dtype=torch.long, device=self.device)
        
        # Sample random noise
        noise = torch.randn_like(latents)
        
        # Add noise to latents
        latents_noisy = self.scheduler.add_noise(latents, noise, t)
        
        # Get LoRA model prediction (with grad for LoRA parameters)
        noise_pred_lora = self.get_noise_preds(latents_noisy, t, text_embeddings, guidance_scale)
        
        # Get pretrained model prediction (without LoRA, stop gradient)
        with torch.no_grad():
            self.unet.disable_adapters()
            noise_pred_pretrained = self.get_noise_preds(latents_noisy.detach(), t, text_embeddings, guidance_scale)
            self.unet.enable_adapters()
        
        # Compute VSD gradient: (ε_θ - ε_φ)
        grad = noise_pred_pretrained - noise_pred_lora
        
        # Convert gradient to loss using the same trick as SDS
        # We want ∂L/∂x_0 = grad
        target = (latents - grad).detach()
        
        # Particle loss: guides the latent towards better samples
        particle_loss = 0.5 * F.mse_loss(latents, target, reduction='sum') / B
        
        # LoRA loss: trains the LoRA parameters to match pretrained model
        # This encourages ε_φ to approximate ε_θ, preventing mode collapse
        lora_loss = F.mse_loss(noise_pred_lora, noise_pred_pretrained, reduction='mean')
        
        # Combined loss
        loss = particle_loss + lora_loss_weight * lora_loss
        
        return loss
    
    @torch.no_grad()
    def invert_noise(self, latents, target_t, text_embeddings, guidance_scale=-7.5, n_steps=10, eta=0.3):
        """
        DDIM Inversion: x0 -> x_t
        
        Inverts clean latents (x0) to noisy latents (x_t) using DDIM inversion.
        
        Args:
            latents: Clean latents x0
            target_t: Target timestep to invert to
            text_embeddings: Text condition
            guidance_scale: CFG scale (typically negative for inversion!)
            n_steps: Number of inversion steps
            eta: Noise level for stochasticity
            
        Returns:
            Inverted noisy latents x_t
        """
        # TODO: (Implement DDIM inversion by yourself — do NOT call built-in inversion helpers):
        # --------------------------------------------------------------------
        # Write your own DDIM inversion loop that maps x0 -> x_t at `target_t`.
        # You may *read* external implementations for reference, but you must
        # NOT call any "invert"/"ddim_invert"/"invert_step" utilities
        # from diffusers or other libraries.
        raise NotImplementedError("TODO: Implement DDIM inversion")
    
    def get_sdi_loss(
        self, 
        latents,                    
        text_embeddings,            
        guidance_scale=7.5,         
        current_iter=0,             
        total_iters=500,            
        inversion_guidance_scale=-7.5,  
        inversion_n_steps=10,       
        inversion_eta=0.3,          
        update_interval=25,        
    ):
        """
        Score Distillation via Inversion (SDI) Loss
        
        Reference: Score Distillation via Reparametrized DDIM (https://arxiv.org/abs/2405.15891)
        
        Key Insight: Instead of using random noise like SDS, SDI uses DDIM inversion
        to get better noise that's consistent with the current latents.
        
        Strategy:
        1. Timestep annealing: t decreases from max_step to min_step during training
        2. Periodically update target via DDIM inversion
        3. Use MSE loss between current latents and cached target
        
        Args:
            latents: Current optimized latents (B, 4, H, W)
            text_embeddings: Concatenated [uncond, cond] embeddings (2*B, seq_len, dim)
            guidance_scale: CFG scale for final denoising step
            current_iter: Current training iteration number
            total_iters: Total number of training iterations
            inversion_guidance_scale: CFG scale for DDIM inversion (typically negative)
            inversion_n_steps: Number of inversion steps from x0 to x_t
            inversion_eta: Stochasticity level for DDIM inversion (0=deterministic)
            update_interval: Update cached target every N iterations
            
        Returns:
            loss: MSE loss between current latents and cached target
        """
        B = latents.shape[0]
        
        # TODO: Create current timestep tensor based on training progress
        # t = ...
        
        # Check if we need to update target
        should_update = (current_iter % update_interval == 0) or not hasattr(self, 'sdi_target')
        
        if should_update:
            with torch.no_grad():
                # Perform DDIM inversion: x0 -> x_t
                latents_noisy = self.invert_noise(
                    latents, t, text_embeddings,
                    guidance_scale=inversion_guidance_scale,
                    n_steps=inversion_n_steps,
                    eta=inversion_eta
                )
                
                # TODO: Predict noise from inverted noisy latents
                # noise_pred = ...
                
                # TODO: Denoise to get target x0 using predicted noise
                # target = ...
                
                # Cache the target
                self.sdi_target = target.detach()
        
        # TODO: Compute MSE loss between current latents and cached target
        # loss = ...
        
        return loss
        
    @torch.no_grad()
    def decode_latents(self, latents):
        """Decode latents to RGB images"""
        latents = 1 / self.vae.config.scaling_factor * latents
        imgs = self.vae.decode(latents).sample
        imgs = (imgs / 2 + 0.5).clamp(0, 1)
        return imgs

    @torch.no_grad()
    def encode_imgs(self, imgs):
        """Encode RGB images to latents"""
        imgs = 2 * imgs - 1
        posterior = self.vae.encode(imgs).latent_dist
        latents = posterior.sample() * self.vae.config.scaling_factor
        return latents