def get_trainer_max_steps(total_training_steps: int, gan_warmup_steps: int, use_gan: bool) -> int:
    """Convert model iterations to Lightning optimizer steps."""
    if total_training_steps <= 0:
        raise ValueError("total_training_steps must be positive")
    if not use_gan:
        return total_training_steps
    if not 0 <= gan_warmup_steps <= total_training_steps:
        raise ValueError("gan_warmup_steps must be between 0 and total_training_steps")

    # One optimizer step before GAN warm-up, then generator + discriminator steps.
    return gan_warmup_steps + 2 * (total_training_steps - gan_warmup_steps)


def get_model_step(global_step: int, gan_warmup_steps: int, use_gan: bool) -> int:
    """Convert Lightning optimizer steps back to completed model iterations."""
    if global_step < 0:
        raise ValueError("global_step must be non-negative")
    if not use_gan or global_step < gan_warmup_steps:
        return global_step
    return global_step - (global_step - gan_warmup_steps) // 2
