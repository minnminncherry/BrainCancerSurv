import cv2
import numpy as np

from utils.core_utils import encoder

class WSISegementation:
    def __init__(self):
        self.level_downsample = None
        self.level_downsamples = None
        self.contour = None
        print("Initialization create patches function ")

    # reduce the resolution of the WSI image
    # seg_range = reduction img resolution, outlier_thred = removing the background value betwwen img and background
    # thred_back = background img pixel value
    def reduce_res_threadshold(self, img, seg_range, outlier_thred=7 ,thred_back=255):
        dimen = img.level_dimensions

        #downsample 
        img = np.array(img.read_region((0,0), seg_range, dimen[seg_range]))
        img_hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)  # Convert to HSV space
        img_med = cv2.medianBlur(img_hsv[:,:,1], outlier_thred)  # Apply median blurring

        # otsu is automatically threadsholding
        img_otsu = cv2.threshold(img_med, 0, thred_back, cv2.THRESH_OTSU+cv2.THRESH_BINARY)
        
        return img_otsu
    
    # Calculate the downsample paramter and filtering the contour and bounging box 
    def downsample_contour(self, slide_img, img_otsu, patch_size=512):
        print("Original Dimension : ", slide_img.level_dimensions)
        filter_params = {
            "a_t": 1,
            "a_h": 1,
            "max_n_holes": 8,
        }
        level_downsamples = []

        
        # level of dimension is largest is full resolution and the 1st is downsamples resolution and second is another downsamples resoultion
        # e.g. ((15349, 23597), (3837, 5899), (1918, 2949))

        # level_downsamples is to determine how much needs to reduce the dimension in the photo like (1.0, 4.0002150702669645, 8.002151186082767)
        dim_0 = slide_img.level_dimensions[0]
        for downsample, dim in zip(slide_img.level_downsamples, slide_img.level_dimensions):
            estimated_downsample = (dim_0[0]/float(dim[0]), dim_0[1]/float(dim[1]))
            level_downsamples.append(estimated_downsample) if estimated_downsample != (downsample, downsample) else level_downsamples.append((downsample, downsample))

        # print("Down Sampling image : ", level_downsamples)
        self.level_downsample = level_downsamples
        self.level_downsamples = level_downsamples

        # tissue segmentation foltering to determines whcih contrours are large enough to be consider tissue and which ones are small artifacts or noise
        mask_level = len(slide_img.level_dimensions) - 1
        # Get the last 
        downsample_x, downsample_y = self.level_downsample[mask_level]
        print(f"downsample X : {downsample_x}, downsample Y : {downsample_y}")

        # Scale the level-0 patch area to the same resolution as the mask.
        scaled_ref_patch_area = int(patch_size**2 / (downsample_x * downsample_y))
        print("Print the Scale reference patch are : ", scaled_ref_patch_area)

        filter_params["a_t"] = filter_params["a_t"] * scaled_ref_patch_area
        filter_params["a_h"] = filter_params["a_h"] * scaled_ref_patch_area

        img_otsu = np.asarray(img_otsu, dtype=np.uint8)

        # boundary using contours and hierarchy parent child relationship
        contours, hierarchy = cv2.findContours(img_otsu, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
        # substitue with 0
        if hierarchy is None:
            return [], []
        hierarchy = np.squeeze(hierarchy, axis=(0,))[:, 2:]
        
        # find and filter contours
        foreground_controur, hole_contours = self.filter_contour(contours, hierarchy, filter_params)
        self.contour = foreground_controur
        return foreground_controur, hole_contours

    def filter_contour(self, contours, hierarchy, filter_params):
        foreground_contours = []
        hole_contours = []
        for i in range(len(contours)):
            # Only process parent contours
            if hierarchy[i][1] != -1:
                continue
            contour = contours[i]
            # Find holes inside this contour
            holes = []
            for j in range(len(contours)):
                if hierarchy[j][1] == i:
                    holes.append(contours[j])
            # Compute actual tissue area
            tissue_area = cv2.contourArea(contour)
            for hole in holes:
                tissue_area -= cv2.contourArea(hole)
            # Ignore small tissue
            if tissue_area < filter_params["a_t"]:
                continue
            foreground_contours.append(contour)
            # Keep only large holes
            valid_holes = []
            for hole in holes:
                if cv2.contourArea(hole) > filter_params["a_h"]:
                    valid_holes.append(hole)
            # Sort by size
            valid_holes.sort(key=cv2.contourArea,reverse=True)
            # Keep largest holes only
            valid_holes = valid_holes[:filter_params.get("max_n_holes", 8)]
            hole_contours.append(valid_holes)

        return foreground_contours, hole_contours
        
    def generate_patch_coord(self, patch_size=512, stride=None, mask_level=None):
        if self.contour is None:
            raise ValueError("Please run downsample_contour() before generate_patch_coord().")
        if self.level_downsamples is None:
            raise ValueError("Missing level downsample values. Please run downsample_contour() first.")

        stride = max(1, int(patch_size // 2)) if stride is None else stride
        mask_level = len(self.level_downsamples) - 1 if mask_level is None else mask_level
        if mask_level >= len(self.level_downsamples):
            mask_level = len(self.level_downsamples) - 1
        downsample_x, downsample_y = self.level_downsamples[mask_level]

        patch_w = max(1, int(round(patch_size / downsample_x)))
        patch_h = max(1, int(round(patch_size / downsample_y)))
        stride_x = max(1, int(round(stride / downsample_x)))
        stride_y = max(1, int(round(stride / downsample_y)))

        patch_coords = []
        for contour in self.contour:
            x, y, w, h = cv2.boundingRect(contour)

            for py in range(y, y + h, stride_y):
                for px in range(x, x + w, stride_x):
                    center = (px + patch_w / 2.0, py + patch_h / 2.0)

                    if cv2.pointPolygonTest(contour, center, False) < 0:
                        continue

                    level0_x = int(round(px * downsample_x))
                    level0_y = int(round(py * downsample_y))
                    patch_coords.append((level0_x, level0_y))

        patch_coords = sorted(set(patch_coords))
        
        print("Patch size : ", patch_size)
        print("Patch stride : ", stride)
        print("Count of Patch Coordination : ", len(patch_coords))

        return patch_coords
    
    # def WSI_format_change_h5(self, patch_coord, output_file, table_name="map_wsi_filepath"):
    #     # Save patch coordinates to a .h5 file for faster downstream loading.
    #     if self.level_downsamples is None:
    #         raise ValueError("Missing level downsample values. Please run downsample_contour() first.")

    #     torch.save(
    #         {
    #             "patch_coords": torch.tensor(patch_coord, dtype=torch.long),
    #             "patch_size": int(patch_size),
    #             "level": 0,
    #             "num_patches": int(len(patch_coord)),
    #         },
    #         str(output_file),
    #     )
    #     print(f"Saved patch coordinates to {output_file} with {len(patch_coord)} tissue patches")
