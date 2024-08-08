import torch
import torch.nn as nn

# 1. attention을 threshold 이하의 값들을 0으로 만들어 이진화
def binarize_attention(attentions, threshold):
    binarized_attentions = torch.where(attentions > threshold, attentions, torch.zeros_like(attentions))
    return binarized_attentions

# 2. 이진화된 attention 맵에서 객체가 있는 좌표를 추출
def extract_object_coordinates(binarized_attentions):
    coordinates = []
    for attention in binarized_attentions:
        nonzero_indices = torch.nonzero(attention)
        min_coords = nonzero_indices.min(dim=0)[0]
        max_coords = nonzero_indices.max(dim=0)[0]
        coordinates.append((min_coords, max_coords))
    return coordinates

# 3. 추출된 좌표를 이용하여 원본 이미지를 cropping
def crop_objects_from_image(image, coordinates):
    cropped_images = []
    image = image.squeeze(0)  # 첫 번째 차원을 제거하여 shape을 (3, 224, 224)로 만듭니다.
    for (min_coords, max_coords) in coordinates:
        cropped_image = image[:, min_coords[0]:max_coords[0]+1, min_coords[1]:max_coords[1]+1]
        cropped_images.append(cropped_image)
    return cropped_images

# 4. Cropped 이미지를 원본 이미지 크기로 업스케일링
def upscale_cropped_images(cropped_images, target_size):
    upscaled_images = []
    for cropped_image in cropped_images:
        upscaled_image = nn.functional.interpolate(cropped_image.unsqueeze(0), size=target_size, mode='bilinear', align_corners=False)
        upscaled_images.append(upscaled_image.squeeze(0))
    return upscaled_images
