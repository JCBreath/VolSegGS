# from segment_anything import SamAutomaticMaskGenerator, sam_model_registry
import numpy as np
import cv2
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import torch
from sklearn.decomposition import PCA
import os
from tqdm import tqdm


class MaskExtractor:
    def __init__(self, use_fast=True, mode='transformers'):
        # registry = sam_model_registry[self.config.sam_model_type]
        # model = registry(checkpoint=self.config.sam_model_ckpt)
        # model = model.to(device=self.config.device)
        # self.model = SamAutomaticMaskGenerator(model=model, **self.config.sam_kwargs)
        self.mode = mode
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        if self.mode == 'transformers':
            print("Running in transformers mode")
            from transformers import pipeline
            self.model = pipeline("mask-generation", model="facebook/sam-vit-huge", device=self.device, use_fast=use_fast)
        if self.mode == 'sam':
            print("Running in sam mode")
            from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
            sam = sam_model_registry["vit_h"](checkpoint="sam_vit_h_4b8939.pth")
            sam.to(device="cuda")
            self.model = SamAutomaticMaskGenerator(sam)


    def set_image(self, image_path, size=None):
        self.image_path = image_path
        self.image = Image.open(image_path)
        if size is not None:
            self.image = self.image.resize(size)
        # self.alpha_mask = self.image.getchannel('A')
        # self.alpha_mask = np.array(self.alpha_mask, dtype=bool)

        # self.alpha_mask = np.stack([self.alpha_mask] * 3, axis=-1)
        # print(self.alpha_mask.shape)
        # self.image = Image.fromarray((self.alpha_mask * 255).astype(np.uint8))
        # print(self.image.size)
        # alpha_mask_img.save(f'trash/{image_path.split("/")[-1][:-4]}_alpha.png')
        background = Image.new("RGBA", self.image.size, (255, 255, 255, 255))
        background.paste(self.image)
        self.image = background
        # self.image.save(f'trash/{image_path.split("/")[-1]}')

    def extract_masks(self):
        # masks = self.model.generate(image)
        # masks = [m['segmentation'] for m in masks] # already as bool
        # masks = sorted(masks, key=lambda x: x.sum())
        img = self.image
        # masks = self.model(img, points_per_side=64, pred_iou_thresh=0.01, stability_score_thresh=0.90)
        if self.mode == 'transformers':
            # masks = self.model(img, points_per_side=32, pred_iou_thresh=0.90, stability_score_thresh=0.90)
            masks = self.model(img, points_per_batch=128, pred_iou_thresh=0.88)
            masks = masks['masks']
        elif self.mode == 'sam':
            masks = self.model.generate(np.array(img.convert("RGB")))
            masks = [m['segmentation'] for m in masks]
            print(masks)
        masks = sorted(masks, key=lambda x: x.sum())
        # cleaned_masks = []
        # for mask in masks:
        #     mask = mask * self.alpha_mask
        #     cleaned_masks.append(mask)
        
        # masks = cleaned_masks
        img = np.array(img)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        print(len(masks))
        return masks

        # Load CLIP model and processor

        clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(self.device)
        clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

        clip_features = []

        for i, mask in enumerate(masks):
            # Crop the image to the masked area
            x, y, w, h = cv2.boundingRect(mask.astype(np.uint8))
            cropped_img = img[y:y+h, x:x+w]

            masked_cropped_img = cv2.bitwise_and(cropped_img, cropped_img, mask=mask[y:y+h, x:x+w].astype(np.uint8))
            masked_cropped_img_pil = Image.fromarray(cv2.cvtColor(masked_cropped_img, cv2.COLOR_BGR2RGB))
            masked_cropped_img_pil.save(f'trash/cropped_image_{i}.png')
            # Process the image and extract features
            inputs = clip_processor(images=masked_cropped_img_pil, return_tensors="pt").to(self.device)
            outputs = clip_model.get_image_features(**inputs)
            print(outputs.shape)
            clip_features.append(outputs.cpu().detach().numpy())

        # Save the features to a file or return them
        np.save('trash/clip_features.npy', np.array(clip_features))

        return masks
    
    def denoise_masks(self, masks):
        all_masks = torch.stack(
            # [torch.from_numpy(_["segmentation"]).to(self.device) for _ in masks]
            [torch.from_numpy(_).to(self.device) for _ in masks]
        )

        eroded_masks = torch.conv2d(
            all_masks.unsqueeze(1).float(),
            torch.full((3, 3), 1.0).view(1, 1, 3, 3).to("cuda"),
            padding=1,
        )
        eroded_masks = (eroded_masks >= 5).squeeze(1)  # (num_masks, H, W)

        return eroded_masks

    def get_mask_scales(self, masks, max_scale=2.0):
        sam_masks = []
        scales = []
        idx = self.image_path.split('/')[-1].split('_')[-1][:-4]
        dir_path = '/'.join(self.image_path.split('/')[:-1])
        point = torch.load(f"{dir_path}/points_{idx}.pth").view(-1,3).cuda()
        alpha = torch.tensor(np.array(Image.open(f"{dir_path}/alpha_{idx}.png"))).view(-1, 3).cuda()[:, 0] / 255.0
        # print(alpha.min(), alpha.max())
        alpha_mask = alpha > 0.1
        # print(point[alpha_mask].shape)
        # exit()
        for i in range(len(masks)):
            curr_mask = masks[i]
            curr_mask = curr_mask.flatten()
            curr_points = point[curr_mask & alpha_mask]
            extent = (curr_points.std(dim=0) * 2).norm()
            x, y, w, h = cv2.boundingRect(masks[i].cpu().numpy().astype(np.uint8))
            if extent.item() < max_scale and w > 10 and h > 10:
                sam_masks.append(curr_mask.reshape(masks[i].shape))
                scales.append(extent.item())
        return sam_masks, scales

    def get_mask_features(self, masks):
        clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(self.device)
        clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        features = []
        for i, mask in enumerate(masks):
            mask = mask.cpu().numpy()
            # mask_pil = Image.fromarray(mask.astype(np.uint8) * 255)
            # mask_pil.save(f'trash/mask_{i}.png')
            masked_img = np.array(self.image)
            masked_img = masked_img[:, :, :3]
            masked_img[~mask] = 255
            # masked_img_pil = Image.fromarray(masked_img)
            # masked_img_pil.save(f'trash/masked_img_{i}.png')
            x, y, w, h = cv2.boundingRect(mask.astype(np.uint8))
            masked_img = masked_img[y:y+h, x:x+w]
            masked_img_pil = Image.fromarray(masked_img)
            # masked_img_pil.save(f'trash/masked_img_{i}_cropped.png')
            inputs = clip_processor(images=masked_img_pil, return_tensors="pt").to(self.device)
            outputs = clip_model.get_image_features(**inputs)
            clip_feature = outputs.cpu().detach().numpy()
            features.append(clip_feature)
        # exit()
        return features


    def combine_masks(self, masks):
        comb_mask = np.zeros(masks[0].shape, dtype=np.int8)
        for idx, mask in enumerate(reversed(masks)):
            comb_mask[mask] = idx
        # print([(comb_mask == idx).sum() for idx in range(len(masks))])
        return comb_mask
    
    def extract_mask_features(self, comb_mask):
        max_value = np.max(comb_mask)
        img = np.array(self.image)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(self.device)
        clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        # feature_img = np.zeros((img.shape[0], img.shape[1], 512), dtype=np.float32)
        features = []
        all_solved = False
        # while not all_solved:
        #     for i in range(max_value+1):
        #         mask = (comb_mask == i)
        #         x, y, w, h = cv2.boundingRect(mask.astype(np.uint8))
        #         print(w, h)
        #         if w < 10 or h < 10:  # Assuming 100 as the threshold for small masks
        #             neighbors = comb_mask[max(0, y-1):min(comb_mask.shape[0], y+h+1), max(0, x-1):min(comb_mask.shape[1], x+w+1)]
        #             unique, counts = np.unique(neighbors, return_counts=True)
        #             neighbor_value = unique[np.argmax(counts)]
        #             comb_mask[mask] = neighbor_value
        #             break
        #     all_solved = True
        for i in range(max_value+1):
            mask = (comb_mask == i)
            x, y, w, h = cv2.boundingRect(mask.astype(np.uint8))
            # if mask.sum() == 0:  # Assuming 100 as the threshold for small masks
            if w < 10 or h < 10:
                features.append(None)
                continue
            
            cropped_img = img[y:y+h, x:x+w]
            masked_cropped_img = cv2.bitwise_and(cropped_img, cropped_img, mask=mask[y:y+h, x:x+w].astype(np.uint8))
            masked_cropped_img_pil = Image.fromarray(cv2.cvtColor(masked_cropped_img, cv2.COLOR_BGR2RGB))
            masked_cropped_img_pil.save(f'trash/cropped_image_{i}.png')
            inputs = clip_processor(images=masked_cropped_img_pil, return_tensors="pt").to(self.device)
            outputs = clip_model.get_image_features(**inputs)
            clip_feature = outputs.cpu().detach().numpy()
            # feature_img[mask] = clip_feature.squeeze()
            features.append(clip_feature)

        if False:
            h, w, c = feature_img.shape
            feature_img_reshaped = feature_img.reshape(-1, c)
            pca = PCA(n_components=3)
            feature_img_pca = pca.fit_transform(feature_img_reshaped)
            feature_img_pca = feature_img_pca.reshape(h, w, 3)
            feature_img_pca_normalized = cv2.normalize(feature_img_pca, None, 0, 255, cv2.NORM_MINMAX)
            feature_img_pca_normalized = feature_img_pca_normalized.astype(np.uint8)
            pca_image = Image.fromarray(feature_img_pca_normalized)
            pca_image.save('trash/feature_img_pca.png')

        return features

if __name__ == '__main__':
    # config = Config()
    extractor = MaskExtractor()
    model_path = 'output/dnerf/fivejets_time_rgb_rep_n100_t10_r800_do_maxinit_f40k'
    dir_path = f'{model_path}/rendered_color_clusters/cluster_1'
    output_dir_path = f'{model_path}/mask_fib'
    file_names = os.listdir(dir_path)

    for file_name in tqdm(file_names):
        if not file_name.endswith('.png'):
            continue
        if not file_name.startswith('image'):
            continue
        if os.path.exists(f'{output_dir_path}/masks_scales_features_{file_name[:-4]}.pt'):
            continue
        print(file_name)
        os.makedirs(f'{output_dir_path}/{file_name[:-4]}', exist_ok=True)
        extractor.set_image(f'{dir_path}/{file_name}')
        masks = extractor.extract_masks()
        exit()
        print(masks[0].shape)
        for idx, mask in enumerate(masks):
            mask_img = Image.fromarray((mask * 255).astype(np.uint8))
            mask_img.save(f'{output_dir_path}/{file_name[:-4]}/mask_{file_name[:-4]}_{idx}.png')
        masks = extractor.denoise_masks(masks)
        for idx, mask in enumerate(masks):
            mask_img = Image.fromarray((mask * 255).cpu().numpy().astype(np.uint8))
            mask_img.save(f'{output_dir_path}/{file_name[:-4]}/denoised_mask_{file_name[:-4]}_{idx}.png')

        masks, scales = extractor.get_mask_scales(masks)
        for scale_idx, scale in enumerate(scales):
            print(scale_idx, scale)

        # features = extractor.get_mask_features(masks)
        features = None

        torch.save({
            'masks': masks,
            'scales': scales,
            'features': features
        }, f'{output_dir_path}/masks_scales_features_{file_name[:-4]}.pt')
        
        # comb_mask = extractor.combine_masks(masks)
        # mask_features = extractor.extract_mask_features(comb_mask)
        # torch.save({'mask_features': mask_features, 'comb_mask': comb_mask}, f'{output_dir_path}/{file_name[:-4]}.pth')
    # comb_mask_normalized = (comb_mask - comb_mask.min()) / (comb_mask.max() - comb_mask.min()) * 255
    # comb_mask_normalized = cv2.normalize(comb_mask_normalized, None, 0, 255, cv2.NORM_MINMAX)
    # colored_mask = cv2.applyColorMap(comb_mask_normalized.astype(np.uint8), cv2.COLORMAP_JET)
    # cv2.imwrite('trash/colored_combined_mask.png', colored_mask)
    # cv2.imwrite('trash/combined_mask.png', comb_mask_normalized.astype(np.uint8))