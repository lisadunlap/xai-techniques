import numpy as np
import cv2
from scipy import ndimage
from PIL import Image

from io import StringIO
from IPython.display import display
from IPython.display import Image

G = [0, 255, 0]
R = [255, 0, 0]

def img_fill(im_in,n):   # n = binary image threshold
    th, im_th = cv2.threshold(im_in, n, 255, cv2.THRESH_BINARY);
     
    # Copy the thresholded image.
    im_floodfill = im_th.copy()
     
    # Mask used to flood filling.
    # Notice the size needs to be 2 pixels than the image.
    h, w = im_th.shape[:2]
    mask = np.zeros((h+2, w+2), np.uint8)
     
    # Floodfill from point (0, 0)
    cv2.floodFill(im_floodfill, mask, (0,0), 255);
     
    # Invert floodfilled image
    im_floodfill_inv = cv2.bitwise_not(im_floodfill)
     
    # Combine the two images to get the foreground.
    fill_image = im_th | im_floodfill_inv
    
    return fill_image 

def convert_to_gray_scale(attributions):
    return np.average(attributions, axis=2)

def linear_transform(attributions, clip_above_percentile=99.9, clip_below_percentile=70.0, low=0.2, plot_distribution=False):
    m = compute_threshold_by_top_percentage(attributions, percentage=100-clip_above_percentile, plot_distribution=plot_distribution)
    e = compute_threshold_by_top_percentage(attributions, percentage=100-clip_below_percentile, plot_distribution=plot_distribution)
    transformed = (1 - low) * (np.abs(attributions) - e) / (m - e) + low
    transformed *= np.sign(attributions)
    transformed *= (transformed >= low)
    transformed = np.clip(transformed, 0.0, 1.0)
    return transformed

def Binarize(attributions, threshold=0.1):
    for t in [.001, .1, .2,]:
        print("len of attr {0} and sum {1} for {2}".format(len(attributions[attributions < t]), len(attributions[attributions > t]), t))
    attributions[attributions < .001] = 0
    attributions[attributions > .001] = 1
    return attributions

def MorphologicalCleanup(attributions, structure=np.ones((4,4))):
    closed = ndimage.grey_closing(attributions, structure=structure)
    opened = ndimage.grey_opening(closed, structure=structure)  
    return opened

def Outlines(attributions, percentage=90,
             connected_component_structure=np.ones((3,3)),
             plot_distribution=True, threshold=0.2):
    # Binarize the attributions mask if not already.
    attributions = Binarize(attributions, threshold=threshold)

    attributions = ndimage.binary_fill_holes(attributions)
  
    # Compute connected components of the transformed mask.
    connected_components, num_cc = ndimage.measurements.label(
          attributions, structure=connected_component_structure)

    # Go through each connected component and sum up the attributions of that
    # component.
    overall_sum = np.sum(attributions[connected_components > 0])
    component_sums = []
    for cc_idx in range(1, num_cc + 1):
        cc_mask = connected_components == cc_idx
        component_sum = np.sum(attributions[cc_mask])
        component_sums.append((component_sum, cc_mask))

    # Compute the percentage of top components to keep.
    sorted_sums_and_masks = sorted(
          component_sums, key=lambda x: x[0], reverse=True)
    sorted_sums = list(zip(*sorted_sums_and_masks))[0]
    cumulative_sorted_sums = np.cumsum(sorted_sums)
    cutoff_threshold = percentage * overall_sum / 100
    cutoff_idx = np.where(cumulative_sorted_sums >= cutoff_threshold)[0][0]

    if cutoff_idx > 2:
        cutoff_idx = 2
  
    # Turn on the kept components.
    border_mask = np.zeros_like(attributions)
    for i in range(cutoff_idx + 1):
        border_mask[sorted_sums_and_masks[i][1]] = 1

    if plot_distribution:
        plt.plot(np.arange(len(sorted_sums)), sorted_sums)
        plt.axvline(x=cutoff_idx)
        plt.show()

    # Hollow out the mask so that only the border is showing.
    eroded_mask = ndimage.binary_erosion(border_mask, iterations=1)
    border_mask[eroded_mask] = 0
  
    return border_mask

def compute_threshold_by_top_percentage(attributions, percentage=60, plot_distribution=True):
    if percentage < 0 or percentage > 100:
        raise ValueError('percentage must be in [0, 100]')
    if percentage == 100:
        return np.min(attributions)
    flat_attributions = attributions.flatten()
    attribution_sum = np.sum(flat_attributions)
    sorted_attributions = np.sort(np.abs(flat_attributions))[::-1]
    cum_sum = 100.0 * np.cumsum(sorted_attributions) / attribution_sum
    threshold_idx = np.where(cum_sum >= percentage)[0][0]
    threshold = sorted_attributions[threshold_idx]
    #if plot_distribution:
    #    raise NotImplementedError 
    return threshold

def polarity_function(attributions, polarity):
    if polarity == 'positive':
        return np.clip(attributions, 0, 1)
    elif polarity == 'negative':
        return np.clip(attributions, -1, 0)
    else:
        raise NotImplementedError

def overlay_function(attributions, image):
    return np.clip(0.7 * image + 0.5 * attributions, 0, 255)

def visualize(attributions, image, positive_channel=G, negative_channel=R, polarity='positive', \
                clip_above_percentile=80, clip_below_percentile=0, morphological_cleanup=False, \
                structure=np.ones((3, 3)), outlines=False, outlines_component_percentage=100, overlay=True, \
                mask_mode=False, plot_distribution=False, threshold = .2):
    if polarity == 'both':
        print("polarity ccant be both")
        raise NotImplementedError

    elif polarity == 'positive':
        attributions = polarity_function(attributions, polarity=polarity)
        channel = positive_channel
    
    # convert the attributions to the gray scale
    attributions = convert_to_gray_scale(attributions)
    attributions = linear_transform(attributions, clip_above_percentile, clip_below_percentile, 0.0, plot_distribution=plot_distribution)
    attributions_mask = attributions.copy()
    #if morphological_cleanup:
    #    raise NotImplementedError
    if outlines:
        #temp_mask = ig_mask/np.max(ig_mask)
        attributions = Outlines(attributions,
                            percentage=outlines_component_percentage,
                            plot_distribution=plot_distribution, threshold=threshold)
    attributions = np.expand_dims(attributions, 2) * channel
    if overlay:
        if mask_mode == False:
            attributions = overlay_function(attributions, image)
        else:
            attributions = np.expand_dims(attributions_mask, 2)
            attributions = np.clip(attributions * image, 0, 255)
            attributions = attributions[:, :, (2, 1, 0)]
    return attributions