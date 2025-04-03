#Import libraries
import cv2
import matplotlib.pyplot as plt
from torch import nn
from PIL import Image as PILImg
from robokit.ObjDetection import GroundingDINOObjectPredictor, SegmentAnythingPredictor
print("Initialize object detectors")
from utils.img_utils import masks_to_bboxes
from robokit.utils import annotate, overlay_masks
from utils.inference_utils import get_features
from utils.inference_utils import compute_similarity
import numpy as np
import torch
from PIL import Image
import os
import json
from tqdm import trange
import math
from utils.inference_utils import FFA_preprocess, get_foreground_mask

#initializing GroundingDino & SAM
gdino = GroundingDINOObjectPredictor(use_vitb=False, threshold=0.5) # we set threshold for GroundingDINO here.
SAM = SegmentAnythingPredictor(vit_model="vit_h")

#Helper Functions
def get_bbox_masks_from_gdino_sam(image_path, gdino, SAM, text_prompt='objects', visualize=False):
    """
    Get bounding boxes and masks from gdino and sam
    @param image_path: the image path
    @param gdino: the model of grounding dino
    @param SAM: segment anything model or its variants
    @param text_prompt: generally 'objects' for object detection of noval objects
    @param visualize: if True, visualize the result
    @return: the bounding boxes and masks of the objects.
    Bounding boxes are in the format of [x_min, y_min, x_max, y_max] and shape of (N, 4).
    Masks are in the format of (N, H, W) and the value is True for object and False for background.
    They are both in the format of torch.tensor.
    """
    image_pil = PILImg.open(image_path).convert("RGB")
    print("GDINO: Predict bounding boxes, phrases, and confidence scores")
    with torch.no_grad():
        bboxes, phrases, gdino_conf = gdino.predict(image_pil, text_prompt)
        w, h = image_pil.size  # Get image width and height
        # Scale bounding boxes to match the original image size
        image_pil_bboxes = gdino.bbox_to_scaled_xyxy(bboxes, w, h)
        print(f"image_pil_bboxes: {image_pil_bboxes}")
        if image_pil_bboxes.numel() == 0:  # Checks if the tensor has zero elements
            print("No bounding boxes found!")


        print("SAM prediction")
        image_pil_bboxes, masks = SAM.predict(image_pil, image_pil_bboxes)
    masks = masks.squeeze(1)
    accurate_bboxs = masks_to_bboxes(masks)  # get the accurate bounding boxes from the masks
    accurate_bboxs = torch.tensor(accurate_bboxs)
    bbox_annotated_pil = None
    if visualize:
        print("Annotate the scaled image with bounding boxes, confidence scores, and labels, and display")
        bbox_annotated_pil = annotate(overlay_masks(image_pil, masks), accurate_bboxs, gdino_conf, phrases)
        plt.imshow(bbox_annotated_pil)
        plt.title("bbox annotated")
        plt.show()
    return accurate_bboxs, masks, bbox_annotated_pil

def get_FFA_feature(img_path, mask, encoder, img_size=448):
    """get FFA for a pair of rgb and mask images"""
    # mask_path = img_path.replace('images', 'masks').replace('.jpg', '.png')
    # mask = Image.open(mask_path)
    # mask = mask.convert('L')
    with open(img_path, 'rb') as f:
        img = Image.open(f)
        img = img.convert('RGB')

    w, h = img.size

    if (img_size is not None) and (min(w, h) > img_size):
        img.thumbnail((img_size, img_size), Image.LANCZOS)
        mask.thumbnail((img_size, img_size), Image.BILINEAR)

        # mask.show()
    else:
        new_w = math.ceil(w / 14) * 14
        new_h = math.ceil(h / 14) * 14
        img = img.resize((new_w, new_h), Image.LANCZOS)
    # mask = mask.resize((16 , 16), Image.BILINEAR)

    with torch.no_grad():
        preprocessed_imgs = FFA_preprocess([img], img_size).to(device)
        mask_size = img_size // 14
        masks = get_foreground_mask([mask], mask_size).to(device)
        emb = encoder.forward_features(preprocessed_imgs)

        grid = emb["x_norm_patchtokens"].view(1, mask_size, mask_size, -1)
        avg_feature = (grid * masks.permute(0, 2, 3, 1)).sum(dim=(1, 2)) / masks.sum(dim=(1, 2, 3)).unsqueeze(-1)

        return avg_feature

def get_object_proposal(image_path, bboxs, masks, tag="mask", ratio=1.0, output_dir='object_proposals', save_segm=False, save_proposal=False):
    """
    Get object proposals from the image according to the bounding boxes and masks.

    @param image_path:
    @param bboxs: numpy array, the bounding boxes of the objects [N, 4]
    @param masks: Boolean numpy array of shape [N, H, W], True for object and False for background
    @param tag: use mask or bbox to crop the object
    @param ratio: ratio to resize the image
    @param save_rois: if True, save the cropped object proposals
    @param output_dir: the folder to save the cropped object proposals
    @return: the cropped object proposals and the object proposals information
    """
    raw_image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
    image_height, image_width = raw_image.shape[:-1]
    scene_name = os.path.basename(image_path).split('.')[0]
    sel_rois = []
    rois = []
    cropped_masks = []
    cropped_imgs = []
    # ratio = 0.25
    if ratio != 1.0:
        scene_image = cv2.resize(raw_image, (int(raw_image.shape[1] * ratio), int(raw_image.shape[0] * ratio)),
                               cv2.INTER_LINEAR)
    else:
        scene_image = raw_image
    for ind in range(len(masks)):
        # bbox
        x0 = int(bboxs[ind][0])
        y0 = int(bboxs[ind][1])
        x1 = int(bboxs[ind][2])
        y1 = int(bboxs[ind][3])

        # load mask
        mask = masks[ind].squeeze(0).cpu().numpy()
        # Assuming `mask` is your boolean numpy array with shape (H, W)
        rle = None
        if save_segm:
            rle = maskUtils.encode(np.asfortranarray(mask.astype(np.uint8)))
            rle['counts'] = rle['counts'].decode('ascii')  # If saving to JSON, ensure counts is a string
        cropped_mask = mask[y0:y1, x0:x1]
        cropped_mask = Image.fromarray(cropped_mask.astype(np.uint8) * 255)
        cropped_masks.append(cropped_mask)
        # show mask
        cropped_img = raw_image[y0:y1, x0:x1]
        cropped_img = Image.fromarray(cropped_img)

        cropped_imgs.append(cropped_img)

        # save bbox
        sel_roi = dict()
        sel_roi['roi_id'] = int(ind)
        sel_roi['mask'] = mask
        #sel_roi['image_id'] = int(scene_name.split('_')[-1])
        sel_roi['bbox'] = [int(x0 * ratio), int(y0 * ratio), int((x1 - x0) * ratio), int((y1 - y0) * ratio)]
        sel_roi['area'] = np.count_nonzero(mask)
        sel_roi['roi_dir'] = os.path.join(output_dir, scene_name, scene_name + '_' + str(ind).zfill(3) + '.png')
        sel_roi['image_dir'] = image_path
        sel_roi['image_width'] = scene_image.shape[1]
        sel_roi['image_height'] = scene_image.shape[0]
        if save_segm:
            sel_roi['segmentation'] = rle  # Add RLE segmentation
        sel_roi['scale'] = int(1 / ratio)
        sel_rois.append(sel_roi)
    if save_proposal:
        with open(os.path.join(output_dir, 'proposals_on_' + scene_name + '.json'), 'w') as f:
            json.dump(sel_rois, f)
    return rois, sel_rois, cropped_imgs, cropped_masks

def visualize_crops(cropped_imgs, cropped_masks):
    num_objects = len(cropped_imgs)
    fig, axes = plt.subplots(num_objects, 2, figsize=(6, 3 * num_objects))

    if num_objects == 1:
        axes = [axes]  # Ensure axes is iterable for a single image case

    for i in range(num_objects):
        # Convert to numpy array if it's in PIL format
        img = np.array(cropped_imgs[i])
        mask = np.array(cropped_masks[i])

        # Display cropped image
        axes[i][0].imshow(img)
        axes[i][0].set_title(f"Cropped Image {i}")
        axes[i][0].axis("off")

        # Display cropped mask
        axes[i][1].imshow(mask, cmap="gray")
        axes[i][1].set_title(f"Cropped Mask {i}")
        axes[i][1].axis("off")

    plt.tight_layout()
    plt.show()

def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)

def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)

def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))

#Load the template image/ Reference image
template_img = 'images/Fatbikes-new/template.png'
image = cv2.imread(template_img)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
plt.figure(figsize=(10,10))
plt.imshow(image)
plt.axis('on')
plt.show()

#Detect with Grounding Dino & Segment with SAM
print("training phase")
accurate_bboxs, masks, vis_img = get_bbox_masks_from_gdino_sam(template_img, gdino, SAM, text_prompt='cycle', visualize=True)
print("bbox shape",accurate_bboxs.shape)
template_box = accurate_bboxs[0].cpu().numpy()
template_mask = masks[0].cpu().numpy()
template_mask = np.expand_dims(template_mask, axis=0)
plt.figure(figsize=(10, 10))
plt.imshow(image)
show_mask(template_mask, plt.gca())
show_box(template_box, plt.gca())
plt.axis('off')
plt.title('SAM+GDino')
plt.show()
print(f' shape is {template_mask.shape}')
print(f' box coords is {template_box}')


# Load dino v2
encoder = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14_reg') # define the DINOv2 version
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
encoder.to(device)
encoder.eval()

img_size = 448
mask_img = (template_mask).astype(np.uint8) * 255  # Convert False to 0, True to 255
mask_img = Image.fromarray(mask_img.squeeze(), 'L')  # 'L' for (8-bit pixels, black and white)
plt.figure(figsize=(10, 10))
plt.imshow(mask_img)
plt.axis('off')
plt.title('Reference Mask')
plt.show()

# use dino v2 to extract features
avg_feature = get_FFA_feature(template_img, mask_img, encoder)
print("Mean of FFA",avg_feature.shape)
object_features = nn.funcget_FFA_featuretional.normalize(avg_feature, dim=1, p=2)


#For single image

# #Load the Test Image
# print("Open the test image and convert to RGB format")
# image_path = "images/Fatbikes-new/test (7).png"
# image_pil = PILImg.open(image_path).convert("RGB")
#
# #Detect & Segment Test image
# accurate_bboxs, masks, vis_img = get_bbox_masks_from_gdino_sam(image_path, gdino, SAM, text_prompt='cycle', visualize=True)
#
# rois, sel_rois, cropped_imgs, cropped_masks = get_object_proposal(image_path, accurate_bboxs, masks, ratio=1.0, output_dir=".", save_proposal=False)
# visualize_crops(cropped_imgs, cropped_masks)
# scene_features = []
# for i in trange(len(cropped_imgs)):
#     img = cropped_imgs[i]
#     mask = cropped_masks[i]
#     ffa_feature= get_features([img], [mask], encoder, device=device, img_size=448)
#     scene_features.append(ffa_feature)
# scene_features = torch.cat(scene_features, dim=0)
# scene_features = nn.functional.normalize(scene_features, dim=1, p=2)
# print(f"Scene Features : {scene_features.shape}")
#
# # Define a similarity threshold (e.g., 0.7)
# threshold = 0.7
#
# # Compute Similarity
# sim_mat = compute_similarity(object_features, scene_features)
# sim_mat = sim_mat.squeeze(-1)  # Remove unnecessary dimensions
# print("sim_mat", sim_mat)
#
# # Find all indices where similarity is above the threshold
# selected_indices = torch.where(sim_mat >= threshold)[0].tolist()
# print("Selected indices:", selected_indices)
#
# # Load and convert the image
# image = cv2.imread(image_path)
# image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#
# # Plot the image
# plt.figure(figsize=(10, 10))
# plt.imshow(image)
#
# # Iterate over all selected objects above the threshold
# for idx in selected_indices:
#     bbox = sel_rois[idx]['bbox']  # Extract bounding box
#     output_mask = sel_rois[idx]['mask']  # Extract mask
#
#     # Convert bbox from [x, y, width, height] to [x_min, y_min, x_max, y_max]
#     x0 = int(bbox[0])
#     y0 = int(bbox[1])
#     x1 = x0 + int(bbox[2])
#     y1 = y0 + int(bbox[3])
#     bbox = [x0, y0, x1, y1]
#
#     # Show the mask and bounding box
#     show_mask(output_mask, plt.gca())
#     show_box(bbox, plt.gca())
#
# plt.axis('off')
# plt.show()



#For all the images in the test directory

# Directory containing test images
image_dir = "images/Fatbikes-new"

# Get list of all images in the directory
image_files = [f for f in os.listdir(image_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]

for image_file in image_files:
    image_path = os.path.join(image_dir, image_file)
    print(f"Processing {image_file}")

    # Open and convert the test image to RGB
    image_pil = PILImg.open(image_path).convert("RGB")

    # Detect & Segment Test Image
    accurate_bboxs, masks, vis_img = get_bbox_masks_from_gdino_sam(
        image_path, gdino, SAM, text_prompt='cycle', visualize=True
    )

    rois, sel_rois, cropped_imgs, cropped_masks = get_object_proposal(
        image_path, accurate_bboxs, masks, ratio=1.0, output_dir=".", save_proposal=False
    )

    visualize_crops(cropped_imgs, cropped_masks)

    scene_features = []
    for i in trange(len(cropped_imgs)):
        img = cropped_imgs[i]
        mask = cropped_masks[i]
        ffa_feature = get_features([img], [mask], encoder, device=device, img_size=448)
        scene_features.append(ffa_feature)

    scene_features = torch.cat(scene_features, dim=0)
    scene_features = torch.nn.functional.normalize(scene_features, dim=1, p=2)
    print(f"Scene Features for {image_file}: {scene_features.shape}")

    # Define a similarity threshold (e.g., 0.7)
    threshold = 0.7

    # Compute Similarity
    sim_mat = compute_similarity(object_features, scene_features)
    sim_mat = sim_mat.squeeze(-1)  # Remove unnecessary dimensions
    print(f"sim_mat for {image_file}:", sim_mat)

    # Find all indices where similarity is above the threshold
    selected_indices = torch.where(sim_mat >= threshold)[0].tolist()
    print(f"Selected indices for {image_file}:", selected_indices)

    # Load and convert the image
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Plot the image
    plt.figure(figsize=(10, 10))
    plt.imshow(image)

    # Iterate over all selected objects above the threshold
    for idx in selected_indices:
        bbox = sel_rois[idx]['bbox']  # Extract bounding box
        output_mask = sel_rois[idx]['mask']  # Extract mask

        # Convert bbox from [x, y, width, height] to [x_min, y_min, x_max, y_max]
        x0 = int(bbox[0])
        y0 = int(bbox[1])
        x1 = x0 + int(bbox[2])
        y1 = y0 + int(bbox[3])
        bbox = [x0, y0, x1, y1]

        # Show the mask and bounding box
        show_mask(output_mask, plt.gca())
        show_box(bbox, plt.gca())

    plt.axis('off')
    plt.show()