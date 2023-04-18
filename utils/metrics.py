from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr


def evaluate(ref, gen):
    psnr_value = psnr(ref, gen)
    ssim_value = ssim(ref, gen, multichannel=True)
    return psnr_value, ssim_value