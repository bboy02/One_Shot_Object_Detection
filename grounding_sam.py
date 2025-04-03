from torch import nn

from GroundingDINO.groundingdino.util.inference import load_model, load_image, predict, annotate
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from segment_anything import SamPredictor, sam_model_registry
from torchvision.ops import box_convert  # Converts bounding box formats
from PIL import Image
import logging
from PIL import Image as PILImg

#detect with GROUNDING DINO & Segment with SAM
def detect_and_segment(image_path, text_prompt, box_threshold=0.35, text_threshold=0.25, visualize=True):
    """
    Detect objects using GroundingDINO and refine segmentation with SAM.

    Args:
        image_path (str): Path to input image.
        text_prompt (str): Object description for detection.
        box_threshold (float): Confidence threshold for object detection.
        text_threshold (float): Confidence threshold for text matching.
        visualize (bool): If True, display the final segmented image.

    Returns:
        dict: Dictionary containing bounding boxes, masks, and segmented image.
    """

    # Load GroundingDINO Model
    model = load_model(
        "GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py",
        "GroundingDINO/weights/groundingdino_swint_ogc.pth"
    )

    # Load Image
    image_source, image = load_image(image_path)

    # GroundingDINO Prediction (Detect Objects)
    boxes, logits, phrases = predict(
        model=model,
        image=image,
        caption=text_prompt,
        box_threshold=box_threshold,
        text_threshold=text_threshold
    )

    # Annotate & Save the Detection Result
    annotated_frame = annotate(image_source=image_source, boxes=boxes, logits=logits, phrases=phrases)
    cv2.imwrite("annotated_groundingdino14.jpg", annotated_frame)

    # Load SAM Model
    sam_checkpoint = "/home/student/pc_deploy/Semester_2/BikeSafeAI/sam_vit_h_4b8939.pth"
    sam = sam_model_registry["vit_h"](checkpoint=sam_checkpoint)

    # Ensure all computations happen on the same device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sam.to(device)  # Move SAM to the correct device

    sam_predictor = SamPredictor(sam)

    # Convert Image to RGB and Set Image for SAM
    image_rgb = cv2.cvtColor(image_source, cv2.COLOR_BGR2RGB)
    sam_predictor.set_image(image_rgb)

    # Convert Bounding Boxes to SAM Format
    H, W, _ = image_source.shape
    boxes = boxes.to(device)  # Move bounding boxes to the same device as SAM
    scale_tensor = torch.tensor([W, H, W, H], device=device)  # Ensure tensor is on the correct device
    boxes_xyxy = box_convert(boxes, in_fmt="cxcywh", out_fmt="xyxy") * scale_tensor

    # Transform Boxes for SAM Input
    transformed_boxes = sam_predictor.transform.apply_boxes_torch(boxes_xyxy, image_source.shape[:2]).to(device)

    # Predict Segmentation Masks using SAM
    masks, _, _ = sam_predictor.predict_torch(
        point_coords=None,
        point_labels=None,
        boxes=transformed_boxes,
        multimask_output=False,
    )

    # Move masks to CPU before converting to NumPy
    masks = masks.cpu().numpy()

    # Apply segmentation masks to all detected objects
    segmented_frame = annotated_frame.copy()  # Start with annotated image
    for i in range(len(boxes)):
        segmented_frame = show_mask(masks[i][0], segmented_frame)

    # Save the segmented image
    segmented_output_path = "segmented_output14.png"
    Image.fromarray(segmented_frame).save(segmented_output_path)

    return {
        "boxes": boxes.cpu().numpy(),
        "masks": masks,
        "segmented_image": segmented_output_path
    }


# Helper function to overlay masks on images
def show_mask(mask, image, random_color=True):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.8])], axis=0)
    else:
        color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])

    h, w = mask.shape[-2:]

    # Ensure mask is a NumPy array
    if isinstance(mask, torch.Tensor):
        mask = mask.cpu().numpy()

    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)  # ✅ Removed unnecessary .cpu()

    # Convert images to PIL format
    annotated_frame_pil = Image.fromarray(image).convert("RGBA")
    mask_image_pil = Image.fromarray((mask_image * 255).astype(np.uint8)).convert("RGBA")

    # Overlay mask on image
    return np.array(Image.alpha_composite(annotated_frame_pil, mask_image_pil))


# Load Tempelate image \ Reference image
result = detect_and_segment(
    image_path="images/2peopleup.png",
    text_prompt="hat"
)

print("Bounding Boxes:", result["boxes"])
print("Segmentation Output Saved:", result["segmented_image"])

#Get template/reference image embeddings:

from utils.inference_utils import get_features, get_features_via_batch_tensor, resize_and_pad, \
    get_weighted_FFA_features
from utils.inference_utils import compute_similarity
from utils.instance_det_dataset import InstanceDataset
import numpy as np
import torch
from PIL import Image
import os
import json
from tqdm import trange, tqdm
from matplotlib import colors
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from utils.instance_det_dataset import BOPDataset
import time
import math
from utils.inference_utils import FFA_preprocess, get_foreground_mask

# use dino v2 to extract features
encoder = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14_reg') # define the DINOv2 version
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
encoder.to(device)
encoder.eval()

img_size = 448


def get_FFA_feature(img_path, mask, encoder, img_size=448):
    """Get FFA for a pair of RGB and mask images"""

    # Convert mask from NumPy array to PIL image (Ensure it's grayscale 'L' format)
    if isinstance(mask, np.ndarray):
        mask = Image.fromarray(mask.astype(np.uint8) * 255, 'L')  # Convert 0/1 to 0/255

    # Load and process the input image
    with open(img_path, 'rb') as f:
        img = Image.open(f).convert('RGB')

    w, h = img.size

    if (img_size is not None) and (min(w, h) > img_size):
        img.thumbnail((img_size, img_size), Image.LANCZOS)
        mask.thumbnail((img_size, img_size), Image.BILINEAR)  # ✅ FIX: mask is now PIL
    else:
        new_w = math.ceil(w / 14) * 14
        new_h = math.ceil(h / 14) * 14
        img = img.resize((new_w, new_h), Image.LANCZOS)

    img.show()
    mask.show()

    with torch.no_grad():
        preprocessed_imgs = FFA_preprocess([img], img_size).to(device)
        mask_size = img_size // 14
        masks = get_foreground_mask([mask], mask_size).to(device)
        emb = encoder.forward_features(preprocessed_imgs)

        grid = emb["x_norm_patchtokens"].view(1, mask_size, mask_size, -1)
        avg_feature = (grid * masks.permute(0, 2, 3, 1)).sum(dim=(1, 2)) / masks.sum(dim=(1, 2, 3)).unsqueeze(-1)

        return avg_feature

template_mask = result["masks"]
template_mask = template_mask[0,0]
mask_img = template_mask.astype(np.uint8) * 255  # Convert False to 0, True to 255
mask_img = Image.fromarray(mask_img, 'L')  # 'L' for (8-bit pixels, black and white)
mask_img.save("reference_mask.png")

template_img = "images/2peopleup.png"

avg_feature = get_FFA_feature(template_img, template_mask, encoder)
object_features = nn.functional.normalize(avg_feature, dim=1, p=2)

#query embeddings
logging.info("Open the image and convert to RGB format")
image_path = "images/cycle_cross.png"
image_pil = PILImg.open(image_path).convert("RGB")
query_result = detect_and_segment(
    image_path=image_path,
    text_prompt="hat"
)
bboxs = query_result["boxes"]


#Proposal embeddings
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

rois, sel_rois, cropped_imgs, cropped_masks = get_object_proposal(image_path, accurate_bboxs, masks, ratio=1.0, output_dir=".", save_proposal=False)
scene_features = []
for i in trange(len(cropped_imgs)):
    img = cropped_imgs[i]
    mask = cropped_masks[i]
    ffa_feature= get_features([img], [mask], encoder, device=device, img_size=448)
    scene_features.append(ffa_feature)
scene_features = torch.cat(scene_features, dim=0)
scene_features = nn.functional.normalize(scene_features, dim=1, p=2)
print(scene_features.shape)
print("SUCCESS")