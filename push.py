import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import copy
import time

from receptive_field import compute_rf_prototype
from helpers import makedir, find_high_activation_crop

import torch.nn.functional as F


# def calculate_regularization_loss(features, attention_map, threshold=0.5):
#     batch_size, num_channels, height, width = features.size()
#     attention_map = attention_map.unsqueeze(1)
#     attention_map = attention_map[:,:,2,:,:]
#     # 배치 크기를 맞추기 위해 복제
#     expanded_attention_map = attention_map.repeat(batch_size // attention_map.size(0), 1, 1, 1)
#     # 어텐션 맵의 채널을 features의 채널 크기로 확장
#     expanded_attention_map = F.interpolate(expanded_attention_map, size=(height, width), mode="nearest")

#     expanded_attention_map = expanded_attention_map.expand(batch_size, num_channels, height, width)
#     # 어텐션 맵에서 높은 값을 가지는 위치의 특징 맵 요소 활성화
#     background_activation = (expanded_attention_map * features).mean(dim=(2, 3))
#     # 배경에서 활성화가 높은 경우에 페널티 부여
#     regularization_loss = (background_activation < threshold).float().mean()
    
#     return regularization_loss

# push each prototype to the nearest patch in the training set
def push_prototypes(dataloader, # pytorch dataloader (must be unnormalized in [0,1])
                    prototype_network_parallel, # pytorch network with prototype_vectors
                    class_specific=True,
                    preprocess_input_function=None, # normalize if needed
                    prototype_layer_stride=1,
                    root_dir_for_saving_prototypes=None, # if not None, prototypes will be saved here
                    epoch_number=None, # if not provided, prototypes saved previously will be overwritten
                    prototype_img_filename_prefix=None,
                    prototype_self_act_filename_prefix=None,
                    proto_bound_boxes_filename_prefix=None,
                    save_prototype_class_identity=True, # which class the prototype image comes from
                    log=print,
                    prototype_activation_function_in_numpy=None):

    prototype_network_parallel.eval()
    log('\tpush')

    start = time.time()
    prototype_shape = prototype_network_parallel.module.prototype_shape
    n_prototypes = prototype_network_parallel.module.num_prototypes
    # saves the closest distance seen so far
    global_min_proto_dist = np.full(n_prototypes, np.inf)
    # saves the patch representation that gives the current smallest distance
    global_min_fmap_patches = np.zeros(
        [n_prototypes,
         prototype_shape[1],
         prototype_shape[2],
         prototype_shape[3]])

    '''
    proto_rf_boxes and proto_bound_boxes column:
    0: image index in the entire dataset
    1: height start index
    2: height end index
    3: width start index
    4: width end index
    5: (optional) class identity
    '''
    if save_prototype_class_identity:
        proto_rf_boxes = np.full(shape=[n_prototypes, 6],
                                    fill_value=-1)
        proto_bound_boxes = np.full(shape=[n_prototypes, 6],
                                            fill_value=-1)
    else:
        proto_rf_boxes = np.full(shape=[n_prototypes, 5],
                                    fill_value=-1)
        proto_bound_boxes = np.full(shape=[n_prototypes, 5],
                                            fill_value=-1)

    if root_dir_for_saving_prototypes != None:
        if epoch_number != None:
            proto_epoch_dir = os.path.join(root_dir_for_saving_prototypes,
                                           'epoch-'+str(epoch_number))
            makedir(proto_epoch_dir)
        else:
            proto_epoch_dir = root_dir_for_saving_prototypes
    else:
        proto_epoch_dir = None

    search_batch_size = dataloader.batch_size

    num_classes = prototype_network_parallel.module.num_classes

    for push_iter, (search_batch_input, search_y) in enumerate(dataloader):
        '''
        start_index_of_search keeps track of the index of the image
        assigned to serve as prototype
        '''
        start_index_of_search_batch = push_iter * search_batch_size

        update_prototypes_on_batch(search_batch_input,
                                   start_index_of_search_batch,
                                   prototype_network_parallel,
                                   global_min_proto_dist,
                                   global_min_fmap_patches,
                                   proto_rf_boxes,
                                   proto_bound_boxes,
                                   class_specific=class_specific,
                                   search_y=search_y,
                                   num_classes=num_classes,
                                   preprocess_input_function=preprocess_input_function,
                                   prototype_layer_stride=prototype_layer_stride,
                                   dir_for_saving_prototypes=proto_epoch_dir,
                                   prototype_img_filename_prefix=prototype_img_filename_prefix,
                                   prototype_self_act_filename_prefix=prototype_self_act_filename_prefix,
                                   prototype_activation_function_in_numpy=prototype_activation_function_in_numpy)


    if proto_epoch_dir != None and proto_bound_boxes_filename_prefix != None:
        np.save(os.path.join(proto_epoch_dir, proto_bound_boxes_filename_prefix + '-receptive_field' + str(epoch_number) + '.npy'),
                proto_rf_boxes)
        np.save(os.path.join(proto_epoch_dir, proto_bound_boxes_filename_prefix + str(epoch_number) + '.npy'),
                proto_bound_boxes)

    log('\tExecuting push ...')
    prototype_update = np.reshape(global_min_fmap_patches,
                                  tuple(prototype_shape))
    prototype_update = torch.tensor(prototype_update, dtype=torch.float32).cuda()
    
    # # 정칙화 손실을 반영하여 프로토타입 업데이트
    # prototype_network_parallel.module.prototype_vectors.data.copy_(
    #     prototype_update - regularization_loss * torch.sign(prototype_update))
    
    prototype_network_parallel.module.prototype_vectors.data.copy_(prototype_update)

    end = time.time()
    log('\tpush time: \t{0}'.format(end -  start))
    # log(f'\tRegularization loss: {regularization_loss.item()}')

    
# update each prototype for current search batch
def update_prototypes_on_batch(search_batch_input,
                               start_index_of_search_batch,
                               prototype_network_parallel,
                               global_min_proto_dist, # this will be updated
                               global_min_fmap_patches, # this will be updated
                               proto_rf_boxes, # this will be updated
                               proto_bound_boxes, # this will be updated
                               class_specific=True,
                               search_y=None, # required if class_specific == True
                               num_classes=None, # required if class_specific == True
                               preprocess_input_function=None,
                               prototype_layer_stride=1,
                               dir_for_saving_prototypes=None,
                               prototype_img_filename_prefix=None,
                               prototype_self_act_filename_prefix=None,
                               prototype_activation_function_in_numpy=None):

    prototype_network_parallel.eval()

    if preprocess_input_function is not None:
        search_batch = preprocess_input_function(search_batch_input)
    else:
        search_batch = search_batch_input

    with torch.no_grad():
        search_batch = search_batch.cuda()
        protoL_input_torch, proto_dist_torch, attention_map = prototype_network_parallel.module.push_forward(search_batch)

    protoL_input_ = np.copy(protoL_input_torch.detach().cpu().numpy())
    proto_dist_ = np.copy(proto_dist_torch.detach().cpu().numpy())

    # 정칙화 손실 계산
    # regularization_loss = calculate_regularization_loss(protoL_input_torch, attention_map)

    del protoL_input_torch, proto_dist_torch

    if class_specific:
        class_to_img_index_dict = {key: [] for key in range(num_classes)}
        for img_index, img_y in enumerate(search_y):
            img_label = img_y.item()
            class_to_img_index_dict[img_label].append(img_index)

    prototype_shape = prototype_network_parallel.module.prototype_shape
    n_prototypes = prototype_shape[0]
    proto_h = prototype_shape[2]
    proto_w = prototype_shape[3]
    max_dist = prototype_shape[1] * prototype_shape[2] * prototype_shape[3]

    for j in range(n_prototypes):
        if class_specific:
            target_class = torch.argmax(prototype_network_parallel.module.prototype_class_identity[j]).item()
            if len(class_to_img_index_dict[target_class]) == 0:
                continue
            proto_dist_j = proto_dist_[class_to_img_index_dict[target_class]][:,j,:,:]
        else:
            proto_dist_j = proto_dist_[:,j,:,:]

        batch_min_proto_dist_j = np.amin(proto_dist_j)
        if batch_min_proto_dist_j < global_min_proto_dist[j]:
            batch_argmin_proto_dist_j = \
                list(np.unravel_index(np.argmin(proto_dist_j, axis=None),
                                      proto_dist_j.shape))
            if class_specific:
                batch_argmin_proto_dist_j[0] = class_to_img_index_dict[target_class][batch_argmin_proto_dist_j[0]]

            img_index_in_batch = batch_argmin_proto_dist_j[0]
            fmap_height_start_index = batch_argmin_proto_dist_j[1] * prototype_layer_stride
            fmap_height_end_index = fmap_height_start_index + proto_h
            fmap_width_start_index = batch_argmin_proto_dist_j[2] * prototype_layer_stride
            fmap_width_end_index = fmap_width_start_index + proto_w

            batch_min_fmap_patch_j = protoL_input_[img_index_in_batch,
                                                   :,
                                                   fmap_height_start_index:fmap_height_end_index,
                                                   fmap_width_start_index:fmap_width_end_index]

            global_min_proto_dist[j] = batch_min_proto_dist_j
            global_min_fmap_patches[j] = batch_min_fmap_patch_j
            
            protoL_rf_info = prototype_network_parallel.module.proto_layer_rf_info
            rf_prototype_j = compute_rf_prototype(search_batch_input.size(2), batch_argmin_proto_dist_j, protoL_rf_info)
            
            original_img_j = search_batch_input[rf_prototype_j[0]]
            original_img_j = original_img_j.cpu().numpy()
            original_img_j = np.transpose(original_img_j, (1, 2, 0))
            original_img_size = original_img_j.shape[0]
            
            rf_img_j = original_img_j[rf_prototype_j[1]:rf_prototype_j[2],
                                      rf_prototype_j[3]:rf_prototype_j[4], :]
            
            proto_rf_boxes[j, 0] = rf_prototype_j[0] + start_index_of_search_batch
            proto_rf_boxes[j, 1] = rf_prototype_j[1]
            proto_rf_boxes[j, 2] = rf_prototype_j[2]
            proto_rf_boxes[j, 3] = rf_prototype_j[3]
            proto_rf_boxes[j, 4] = rf_prototype_j[4]
            if proto_rf_boxes.shape[1] == 6 and search_y is not None:
                proto_rf_boxes[j, 5] = search_y[rf_prototype_j[0]].item()

            proto_dist_img_j = proto_dist_[img_index_in_batch, j, :, :]
            if prototype_network_parallel.module.prototype_activation_function == 'log':
                proto_act_img_j = np.log((proto_dist_img_j + 1) / (proto_dist_img_j + prototype_network_parallel.module.epsilon))
            elif prototype_network_parallel.module.prototype_activation_function == 'linear':
                proto_act_img_j = max_dist - proto_dist_img_j
            else:
                proto_act_img_j = prototype_activation_function_in_numpy(proto_dist_img_j)
            upsampled_act_img_j = cv2.resize(proto_act_img_j, dsize=(original_img_size, original_img_size),
                                             interpolation=cv2.INTER_CUBIC)
            proto_bound_j = find_high_activation_crop(upsampled_act_img_j)
            proto_img_j = original_img_j[proto_bound_j[0]:proto_bound_j[1],
                                         proto_bound_j[2]:proto_bound_j[3], :]

            proto_bound_boxes[j, 0] = proto_rf_boxes[j, 0]
            proto_bound_boxes[j, 1] = proto_bound_j[0]
            proto_bound_boxes[j, 2] = proto_bound_j[1]
            proto_bound_boxes[j, 3] = proto_bound_j[2]
            proto_bound_boxes[j, 4] = proto_bound_j[3]
            if proto_bound_boxes.shape[1] == 6 and search_y is not None:
                proto_bound_boxes[j, 5] = search_y[rf_prototype_j[0]].item()

            if dir_for_saving_prototypes is not None:
                if prototype_self_act_filename_prefix is not None:
                    np.save(os.path.join(dir_for_saving_prototypes,
                                         prototype_self_act_filename_prefix + str(j) + '.npy'),
                            proto_act_img_j)
                if prototype_img_filename_prefix is not None:
                    plt.imsave(os.path.join(dir_for_saving_prototypes,
                                            prototype_img_filename_prefix + '-original' + str(j) + '.png'),
                               original_img_j,
                               vmin=0.0,
                               vmax=1.0)
                    rescaled_act_img_j = upsampled_act_img_j - np.amin(upsampled_act_img_j)
                    rescaled_act_img_j = rescaled_act_img_j / np.amax(rescaled_act_img_j)
                    heatmap = cv2.applyColorMap(np.uint8(255*rescaled_act_img_j), cv2.COLORMAP_JET)
                    heatmap = np.float32(heatmap) / 255
                    heatmap = heatmap[...,::-1]
                    overlayed_original_img_j = 0.5 * original_img_j + 0.3 * heatmap
                    plt.imsave(os.path.join(dir_for_saving_prototypes,
                                            prototype_img_filename_prefix + '-original_with_self_act' + str(j) + '.png'),
                               overlayed_original_img_j,
                               vmin=0.0,
                               vmax=1.0)
                    
                    if rf_img_j.shape[0] != original_img_size or rf_img_j.shape[1] != original_img_size:
                        plt.imsave(os.path.join(dir_for_saving_prototypes,
                                                prototype_img_filename_prefix + '-receptive_field' + str(j) + '.png'),
                                   rf_img_j,
                                   vmin=0.0,
                                   vmax=1.0)
                        overlayed_rf_img_j = overlayed_original_img_j[rf_prototype_j[1]:rf_prototype_j[2],
                                                                      rf_prototype_j[3]:rf_prototype_j[4]]
                        plt.imsave(os.path.join(dir_for_saving_prototypes,
                                                prototype_img_filename_prefix + '-receptive_field_with_self_act' + str(j) + '.png'),
                                   overlayed_rf_img_j,
                                   vmin=0.0,
                                   vmax=1.0)
                    
                        plt.imsave(os.path.join(dir_for_saving_prototypes,
                                                prototype_img_filename_prefix + str(j) + '.png'),
                                   proto_img_j,
                                   vmin=0.0,
                                   vmax=1.0)
    # return regularization_loss
