from abc import ABC, abstractmethod
import numpy as np
import cv2
from skimage.morphology import skeletonize as ski_skeletonize
import os
from scipy import ndimage
from scipy.ndimage import rotate, grey_dilation

class FingerprintAnalysisTemplate(ABC):

    def analyze_fingerprint(self) -> None:
        self.normalize()
        self.calculate_orientation()
        self.calculate_frequency()
        self.apply_gabor_filters()
        self.skeletonize()
        self.calculate_minutiae_weights()
        self.select_best_region()
        self.draw_ridges_count_on_region()
        
    @abstractmethod
    def normalize(self):
        pass

    @abstractmethod
    def calculate_orientation(self):
        pass

    @abstractmethod
    def calculate_frequency(self):
        pass

    @abstractmethod
    def apply_gabor_filters(self):
        pass

    @abstractmethod
    def skeletonize(self):
        pass

    @abstractmethod
    def calculate_minutiae_weights(self):
        pass

    @abstractmethod
    def select_best_region(self):
        pass

    @abstractmethod
    def draw_ridges_count_on_region(self):
        pass

class ConcreteFingerprintAnalysis1(FingerprintAnalysisTemplate):
    
    def __init__(self):
        # Initialize any class-wide variables here, if necessary
        pass

    def analyze_fingerprint(self, image):
        # This method uses the template method pattern to run the algorithm steps
        print(image.shape)
        cropped_img = image[:-32, :]  
        mask, normalized_img = self.normalize(cropped_img)

        orientation = self.calculate_orientation(normalized_img)

        frequency_masked = self.calculate_frequency(normalized_img, orientation, mask)

        gabor_filtered_img = self.apply_gabor_filters(normalized_img, frequency_masked, orientation)

        thin_image = self.skeletonize(gabor_filtered_img)
        minutiae_weights_image = self.calculate_minutiae_weights(thin_image)

        block_size = 15 # 120/8
        best_region = self.select_best_region(thin_image, minutiae_weights_image, block_size, mask)
        output_image = self.draw_ridges_count_on_region(best_region, image, thin_image, block_size)
        return output_image

    def normalize(self, image, block_size=16, std_threshold=0.05):
        img_normalized = (image - np.min(image)) / (np.max(image) - np.min(image))
        rows, cols = img_normalized.shape
        mask = np.zeros((rows, cols), dtype=bool)
        for i in range(0, rows, block_size):
            for j in range(0, cols, block_size):
                block = img_normalized[i:i + block_size, j:j + block_size]
                std_dev = np.std(block)
                mask[i:i + block_size, j:j + block_size] = std_dev > std_threshold
        masked_image = img_normalized[mask]
        img_norm_masked = (img_normalized - np.mean(masked_image)) / np.std(masked_image)
        return mask, img_norm_masked

    def calculate_orientation(self, normalized_img, grad_rate=1, block_rate=7, orient_smooth_rate=7):
        gradient_kernel = cv2.getGaussianKernel(np.int_(6 * grad_rate + 1), grad_rate)
        gradient_matrix = gradient_kernel * gradient_kernel.T
        gy, gx = np.gradient(gradient_matrix)
        gradient_x = ndimage.convolve(normalized_img, gx, mode='constant', cval=0.0)
        gradient_y = ndimage.convolve(normalized_img, gy, mode='constant', cval=0.0)
        Gxx, Gyy, Gxy = gradient_x ** 2, gradient_y ** 2, gradient_x * gradient_y
        block_kernel = cv2.getGaussianKernel(6 * block_rate, block_rate)
        block_matrix = block_kernel * block_kernel.T
        Gxx = ndimage.convolve(Gxx, block_matrix, mode='constant', cval=0.0)
        Gyy = ndimage.convolve(Gyy, block_matrix, mode='constant', cval=0.0)
        Gxy = 2 * ndimage.convolve(Gxy, block_matrix, mode='constant', cval=0.0)
        determinant = np.sqrt((Gxx - Gyy) ** 2 + 4 * Gxy ** 2)
        determinant = np.where(determinant == 0, np.finfo(float).eps, determinant)
        determinant += np.finfo(float).eps
        sin_2theta = Gxy / determinant
        cos_2theta = (Gxx - Gyy) / determinant
        if orient_smooth_rate:
            smooth_kernel = cv2.getGaussianKernel(np.int_(6 * orient_smooth_rate + 1), orient_smooth_rate)
            smooth_matrix = smooth_kernel * smooth_kernel.T
            cos_2theta = ndimage.convolve(cos_2theta, smooth_matrix, mode='constant', cval=0.0)
            sin_2theta = ndimage.convolve(sin_2theta, smooth_matrix, mode='constant', cval=0.0)
        orientation = np.pi / 2 + np.arctan2(sin_2theta, cos_2theta) / 2
        return orientation

    def calculate_frequency(self, normalized_img, orientation, mask, block_size=38, window_size=5, min_wavelength=5, max_wavelength=20):
        frequencies = np.zeros_like(normalized_img)

        for row in range(0, frequencies.shape[0] - block_size, block_size):
            for col in range(0, frequencies.shape[1] - block_size, block_size):
                block_image = normalized_img[row:row + block_size, col:col + block_size]
                block_orientation = orientation[row:row + block_size, col:col + block_size]
                frequencies[row:row + block_size, col:col + block_size] = self.calculate_block_frequencies(block_image,
                                                                                                    block_orientation,
                                                                                                    window_size,
                                                                                                    min_wavelength,
                                                                                                    max_wavelength)
        masked_frequencies = frequencies * mask
        non_zero_elements = masked_frequencies[mask > 0]
        mean_frequency = np.mean(non_zero_elements)
        frequency_masked = mean_frequency * mask

        return frequency_masked
      
    def apply_gabor_filters(self, image, frequency, orientation, threshold=-2, kx=0.65, ky=0.65, angle_increment=3):
        image = image.astype(np.float64)
        rows, cols = image.shape
        filtered_image = np.zeros((rows, cols))

        non_zero_freqs = frequency.ravel()[frequency.ravel() > 0]
        rounded_freqs = np.round(non_zero_freqs * 100) / 100
        unique_freqs = np.unique(rounded_freqs)

        sigma_x = 1 / unique_freqs[0] * kx
        sigma_y = 1 / unique_freqs[0] * ky
        filter_size = np.int_(np.round(3 * np.max([sigma_x, sigma_y])))

        x, y = np.meshgrid(np.arange(-filter_size, filter_size + 1), np.arange(-filter_size, filter_size + 1))
        exponent = ((x / sigma_x) ** 2 + (y / sigma_y) ** 2) / 2
        reference_filter = np.exp(-exponent) * np.cos(2 * np.pi * unique_freqs[0] * x)
        angle_range = int(180 / angle_increment)
        gabor_filters = np.array(
            [rotate(reference_filter, -(o * angle_increment + 90), reshape=False) for o in range(angle_range)])

        max_size = filter_size
        valid_rows, valid_cols = np.where(frequency > 0)
        valid_indices = np.where((valid_rows > max_size) & (valid_rows < rows - max_size) & (valid_cols > max_size) & (
                valid_cols < cols - max_size))[0]
        max_orient_index = int(np.round(180 / angle_increment))
        orient_index = np.round(orientation / np.pi * 180 / angle_increment).astype(np.int32)
        orient_index[orient_index < 1] += max_orient_index
        orient_index[orient_index > max_orient_index] -= max_orient_index

        for k in valid_indices:
            r = valid_rows[k]
            c = valid_cols[k]
            img_block = image[r - filter_size:r + filter_size + 1, c - filter_size:c + filter_size + 1]
            filtered_image[r, c] = np.sum(img_block * gabor_filters[orient_index[r, c] - 1])

        binary_image = (filtered_image < threshold) * 255

        gabor_filtered_img = (255 - binary_image).astype(np.uint8)
        return gabor_filtered_img

    def skeletonize(self, img):

        binary_image = np.zeros_like(img)
        binary_image[img == 0] = 1.0

        skeleton = ski_skeletonize(binary_image)

        output_img = (1 - skeleton) * 255.0
        skeleton = output_img.astype(np.uint8)

        return skeleton

    def calculate_minutiae_weights(self, image):
        binary_image = (image == 0).astype(np.int8)
        minutiae_weights_array = np.zeros_like(image, dtype=np.float_)

        for col in range(1, image.shape[1] - 1):
            for row in range(1, image.shape[0] - 1):
                minutiae = self.detect_minutiae(binary_image, row, col)
                if minutiae == "bifurcation":
                    minutiae_weights_array[row, col] += 1.0
                elif minutiae == "termination":
                    minutiae_weights_array[row, col] += 2.0

        return minutiae_weights_array
    
    def select_best_region(self, thin_image, minutiae_weights_image, block_size, mask):

        best_region = None
        block_minutiaes_weight = float('inf')
        rows, cols = thin_image.shape
        shape_block = block_size * 8
        window = (shape_block // block_size) - 1
        for col in range(1, cols - window):
            for row in range(1, rows - window):
                mask_slice = mask[col * block_size:col * block_size + shape_block,
                            row * block_size:row * block_size + shape_block]
                mask_flag = np.sum(mask_slice)
                if mask_flag == shape_block * shape_block:
                    number_vision_problems = np.sum(minutiae_weights_image[col * block_size:col * block_size + shape_block,
                                                    row * block_size:row * block_size + shape_block])
                    if number_vision_problems <= block_minutiaes_weight:
                        block_minutiaes_weight = number_vision_problems
                        best_region = [col * block_size, col * block_size + shape_block, row * block_size,
                                    row * block_size + shape_block]
        if best_region:
            return best_region

    def draw_ridges_count_on_region(self, region, input_image, thin_image, block_size):
   
        output_image = cv2.cvtColor(input_image.copy(), cv2.COLOR_GRAY2RGB)
        if region is None:
            return output_image
        region_copy = (thin_image[region[0]: region[1], region[2]: region[3]]).copy()
        line_count, line_type = self.count_lines(region_copy)
        text_y = (region[3] - region[2]) // 2 + region[2]
        if line_type == "main_diagonal":
            text_x = (region[1] - region[0]) // 3 + region[0]
            self.draw_diagonal(output_image, region[0], region[2], region[1], region[3], line_count, block_size, text_y, text_x,
                        (0, 255, 255))
        elif line_type == "secondary_diagonal":
            text_x = 2 * (region[1] - region[0]) // 3 + region[0]
            self.draw_diagonal(output_image, region[0], region[3], region[1], region[2], line_count, block_size, text_y, text_x,
                        (0, 255, 255))
        cv2.rectangle(output_image, (region[2], region[0]), (region[3], region[1]), (0, 0, 255), 1)
        return output_image
    
    # Maybe need to move to supperclass  
    def calculate_block_frequencies(self, block_image, block_orientation, window_size, min_wavelength, max_wavelength):

        cos_orient = np.mean(np.cos(2 * block_orientation))
        sin_orient = np.mean(np.sin(2 * block_orientation))
        orientation = np.arctan2(sin_orient, cos_orient) / 2

        rotated_image = rotate(block_image, orientation / np.pi * 180 + 90, axes=(1, 0), reshape=False,
                            order=3, mode='nearest')
        crop_size = np.sqrt(block_image.shape[0] * block_image.shape[1] / 2).astype(np.int_)
        offset = (block_image.shape[0] - crop_size) // 2
        cropped_image = rotated_image[offset:offset + crop_size, offset:offset + crop_size]

        projection = np.sum(cropped_image, axis=0)
        dilation = grey_dilation(projection, window_size, structure=np.ones(window_size))
        noise = np.abs(dilation - projection)
        peak_thresh = 2
        max_pts = (noise < peak_thresh) & (projection > np.mean(projection))
        max_ind = np.where(max_pts)
        num_peaks = len(max_ind[0])
        if num_peaks < 2:
            return np.zeros(block_image.shape)
        else:
            wavelength = (max_ind[0][-1] - max_ind[0][0]) / (num_peaks - 1)
            if min_wavelength <= wavelength <= max_wavelength:
                return 1 / np.double(wavelength) * np.ones(block_image.shape)
            else:
                return np.zeros(block_image.shape)

    def detect_minutiae(self, image, row, col):
    
        if image[row][col] == 1:
            kernel = [(-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1), (-1, -1)]
            values = []
            for k, l in kernel:
                values.append(image[row + l][col + k])
            crossings = sum(abs(values[k] - values[k + 1]) for k in range(len(values) - 1)) // 2
            if crossings == 1:
                return "termination"
            if crossings == 3:
                return "bifurcation"
        return "none"

    def count_lines(self, image):
    
        kernel = np.ones((3, 3), dtype=np.uint8)
        eroded_image = cv2.erode(image, kernel)
        binary_image = cv2.threshold(eroded_image, 127, 255, cv2.THRESH_BINARY)[1]
        main_diagonal = self.count_diagonal_lines(binary_image)
        secondary_diagonal = self.count_diagonal_lines(np.fliplr(binary_image))
        return (secondary_diagonal, "secondary_diagonal") if main_diagonal <= secondary_diagonal else (
            main_diagonal, "main_diagonal")

    def count_diagonal_lines(self, image):
    
        is_white = True
        counter = 0
        for i in range(image.shape[0]):
            if image[i][i] == 0 and is_white:
                counter += 1
                is_white = False
            if image[i][i] == 255:
                is_white = True
        return counter

    def draw_diagonal(self, image, start_i, start_j, end_i, end_j, line_count, block_size, text_y, text_x, color):
    
        cv2.line(image, (start_j, start_i), (end_j, end_i), (0, 0, 255), 1)
        cv2.putText(image, f' {line_count}', (text_y, text_x - block_size), cv2.FONT_HERSHEY_TRIPLEX, 0.5, color, 1)
        cv2.putText(image, "ridges", (text_y, text_x), cv2.FONT_HERSHEY_TRIPLEX, 0.5, color, 1)


if __name__ == '__main__':
    input_path = './all_png_files/'
    output_path = './all_png_files_out/'

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    analysis = ConcreteFingerprintAnalysis1()

    for img_name in os.listdir(input_path):
        img_dir = os.path.join(input_path, img_name)
        greyscale_image = cv2.imread(img_dir, 0)  

        if greyscale_image is None:
            continue

        print(img_name)

        output_image = analysis.analyze_fingerprint(greyscale_image)

        cv2.imwrite(os.path.join(output_path, img_name), output_image)