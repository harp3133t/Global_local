import os
import torch
import torch.utils.data
import numpy as np
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import argparse
import glob
from PIL import Image
import cv2
import matplotlib.pyplot as plt
from preprocess import mean, std, preprocess_input_function
import model
import random
# from settings import *
from util.deletion_auc import CausalMetric, auc
from util.iou import iou_metric
from tqdm.auto import tqdm
import settings_CUB_DOG

parser = argparse.ArgumentParser()
parser.add_argument('-gpuid', nargs=1, type=str, default='0')
parser.add_argument('-dataset',)
parser.add_argument('-exp', type=str, default='1000')
args = parser.parse_args()


os.environ['CUDA_VISIBLE_DEVICES'] = '0'
print("GPU ID:", os.environ['CUDA_VISIBLE_DEVICES'])



base_architecture = 'vgg19'
torch.manual_seed(41)
random.seed(41)
np.random.seed(41)


# 필요한것
# 1. 입력 이미지의 Prototype 추출
# 2. 추출한 Prototype을 IOU 계산하는 함수
# 3. 얼마나 맞는지
num_classes = settings_CUB_DOG.num_classes
img_size = settings_CUB_DOG.img_size
add_on_layers_type = settings_CUB_DOG.add_on_layers_type
prototype_shape = settings_CUB_DOG.prototype_shape
prototype_activation_function = settings_CUB_DOG.prototype_activation_function

# construct the model
ppnet = model.construct_STProtoPNet(base_architecture=base_architecture,
                                    pretrained=True, img_size=img_size,
                                    prototype_shape=prototype_shape,
                                    num_classes=num_classes,
                                    prototype_activation_function=prototype_activation_function,
                                    add_on_layers_type=add_on_layers_type)
ppnet = ppnet.cuda()
class_specific = True


# pretrained model
checkpoint_path = glob.glob('./saved_models/20240623_1155/'+base_architecture+'/'+args.exp+'/20_9push0.7721.pth')#"80nopush0.8024_vgg19_40p.pth"
ppnet.load_state_dict(torch.load(checkpoint_path[0]))
# ppnet = torch.load(checkpoint_path[0])
model = torch.nn.DataParallel(ppnet)
# print(ppnet._metadata.keys())
# model = ppnet
model.eval()


# all datasets
num_workers = 0  # 20, 8
normalize = transforms.Normalize(mean=mean, std=std)
transform = transforms.Compose([
            transforms.Resize(size=(img_size, img_size)),
            transforms.ToTensor(),
            normalize,
            ])


data_path = '../data/'


test_batch_size = 1
test_dir = '../data_legacy/full/full_test/'
test_dataset = datasets.ImageFolder(
    test_dir,
    transform,
    )
test_loader = torch.utils.data.DataLoader(
    test_dataset, batch_size=test_batch_size, shuffle=False,
    num_workers=num_workers, pin_memory=False)

GT_PATH =  '../data_legacy/se gmentations'#data_path + 'seg/'

deletion = CausalMetric(model, 'del', step=224 * 8, substrate_fn=torch.zeros_like)

del_all = []
ch_all = []
iou_all = []
oirr_all = []

import torch.nn.functional as F
def model_forward(model, x):
    attentions = model.module.get_attention(x)
    distances1, conv_features = model.module.prototype_distances(x, attentions)
     # global min pooling
    min_distances1 = -F.max_pool2d(-distances1,
                                    kernel_size=(distances1.size()[2],
                                                distances1.size()[3]))
    # print(min_distances1.shape)
    # print(model.module.num_prototypes)
    min_distances1 = min_distances1.view(-1, model.module.num_prototypes)
    # print(min_distances1.shape)
    
    prototype_activations1 = model.module.distance_2_similarity(min_distances1)
    return prototype_activations1


def return_proto(model, x):
    with torch.no_grad():
        x = x.cuda()
        # this computation currently is not parallelized
        protoL_input_torch, proto_dist_torch, attentions = model.module.push_forward(x)
    protoL_input_ = np.copy(protoL_input_torch.detach().cpu().numpy())
    proto_dist_ = np.copy(proto_dist_torch.detach().cpu().numpy())
    del protoL_input_torch, proto_dist_torch, attentions
    


for i, (img_name, label) in enumerate(tqdm(test_dataset.imgs)):

    img_ori_PIL = Image.open(img_name)
    img_ori_PIL = img_ori_PIL.convert('RGB')
    img = transform(img_ori_PIL)

    gt_seg_path = os.path.join(GT_PATH, '/'.join(img_name.split('/')[-2:])).replace('.jpg', '.png')
    # print(gt_seg_path)
    gt_mask = cv2.imread(gt_seg_path)[:, :, 0]
    gt_mask[gt_mask != 0] = 1

    img_np = img.cpu().numpy().transpose(1, 2, 0).astype(np.float32)
    img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min())

    input = img.unsqueeze(0).cuda()
    target = torch.tensor(label).cuda()

    with torch.no_grad():
        prototype_activations1 = model_forward(model,input)#model(input)
    # print('model.module.num_prototypes:',model.module.num_prototypes)
    # print('model.module.num_classes:',model.module.num_classes)
    num_proto_per_class = int(model.module.num_prototypes/model.module.num_classes)
    prototypes_of_correct_class = torch.t(model.module.prototype_class_identity[:, label]).cuda()
    prototypes_of_correct_class = prototypes_of_correct_class[:2000]
    # print('prototype_activations1:',prototype_activations1.shape)
    # print('prototypes_of_correct_class:',prototypes_of_correct_class.shape)


    proto_act_img_0 = prototype_activations1[:, prototypes_of_correct_class == 1].squeeze().detach().cpu().numpy()
    # print('proto_act_img_0:',proto_act_img_0.shape)
    # proto_act_img_0 = proto_act_img_0.mean(0)
    proto_act = proto_act_img_0 
    # print('type(proto_act):',type(proto_act))
    # print('proto_act.shape:',proto_act.shape)

    heatmap_deletion = cv2.resize(proto_act, dsize=(img_size, img_size), interpolation=cv2.INTER_CUBIC)
    heatmap_deletion = (heatmap_deletion - heatmap_deletion.min()) / (heatmap_deletion.max() - heatmap_deletion.min())
    # print('heatmap_deletion:',heatmap_deletion.shape)

    cv2.imwrite('heatmap_deletion.jpg', heatmap_deletion*255)

    del_score = deletion.single_run(input, heatmap_deletion, verbose=0)   # verbose=2
    del_auc = auc(del_score)
    del_all.append(del_auc)

    heatmap = cv2.resize(proto_act, dsize=(img_ori_PIL.size[0], img_ori_PIL.size[1]), interpolation=cv2.INTER_CUBIC)  # original image resolution
    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())
    
    cv2.imwrite('heatmap.jpg', heatmap*255)
    
    # print('gt_mask:',gt_mask.shape)
    # print('heatma:',heatmap.shape)

    ch = heatmap[gt_mask == 1].sum() / (gt_mask == 1).sum()
    oirr = (heatmap[gt_mask == 0].sum() / (gt_mask == 0).sum()) / (heatmap[gt_mask == 1].sum() / (gt_mask == 1).sum())
    iou = iou_metric(heatmap, gt_mask)

    ch_all.append(ch)
    oirr_all.append(oirr)
    iou_all.append(iou)

    if i % 100 == 0:
        print(i, len(test_dataset.imgs),
            "DAUC:", np.round(sum(del_all)/len(del_all), 4),
            "CH:", np.round(sum(ch_all)/len(ch_all), 4),
            "OIRR:", np.round(sum(oirr_all)/len(oirr_all), 4),
            "IoU:", np.round(sum(iou_all)/len(iou_all), 4),
            )


print('Number of samples:', len(del_all))
print('')
print('Mean DAUC:', np.round(sum(del_all)/len(del_all), 4))
print('Mean CH:', np.round(sum(ch_all)/len(ch_all), 4))
print('Mean OIRR:', np.round(sum(oirr_all)/len(oirr_all), 4))
print('Mean IoU:', np.round(sum(iou_all)/len(iou_all), 4))
