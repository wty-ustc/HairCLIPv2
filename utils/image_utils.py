import matplotlib.pyplot as plt
import torch
import numpy as np
import cv2

def process_tensor_dimensions(input_tensor):
    if input_tensor.dim() == 4:
        output_tensor = input_tensor
    elif input_tensor.dim() == 3:
        output_tensor = input_tensor.unsqueeze(1)
    elif input_tensor.dim() == 2:
        output_tensor = input_tensor.unsqueeze(0).unsqueeze(0)
    else:
        raise ValueError("Unsupported number of dimensions. Supported dimensions are 2, 3, or 4.")
    return output_tensor

def process_display_input(input_data):
    if isinstance(input_data, np.ndarray):
        output_data = input_data
    elif isinstance(input_data, torch.Tensor):
        output_data = ((process_tensor_dimensions(input_data)[0]+1) / 2).permute(1,2,0).cpu().detach()
        if output_data.shape[-1] == 1:
            output_data = output_data.repeat(1,1,3)
        output_data = 255.0 * output_data.numpy()
        output_data = np.clip(output_data, 0, 255).astype(np.uint8)
    else:
        raise ValueError("Unsupported input type. Supported types are numpy.ndarray and torch.Tensor.")
    return output_data

def display_image_list(img_list):
    fig_final, ax_final = plt.subplots(1, len(img_list), figsize=(5*len(img_list), 5))
    for idx in range(len(img_list)):
        ax_final[idx].imshow(process_display_input(img_list[idx]))

def dliate_erode(img, kernel):
    er_k = kernel
    di_k = kernel
    erode_kernel = cv2.getStructuringElement(cv2.MORPH_RECT,((er_k//2), (er_k//2)))
    dilate_kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(di_k, di_k))
    img_f = cv2.dilate(img, dilate_kernel)
    img_f = cv2.erode(img_f, erode_kernel)
    return img_f



