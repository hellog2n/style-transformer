from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

def to_numpy(data):
    return data.detach().cpu().numpy().transpose(1, 2, 0)

def evaluate_batch(ref, gen):
    psnr_value_list = []
    ssim_value_list = []
    for r, g in zip(ref, gen):
        psnr, ssim = evaluate_sample(r, g)
        psnr_value_list.append(psnr)
        ssim_value_list.append(ssim)
    return psnr_value_list, ssim_value_list

def evaluate_sample(r, g):
    r = to_numpy(r)
    g = to_numpy(g)
    psnr_value = psnr(r, g)
    ssim_value = ssim(r, g, multichannel=True)
    return psnr_value, ssim_value