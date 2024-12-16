from mpi4py import MPI
import numpy as np
from scipy.ndimage import convolve
from PIL import Image
import os

# Define the Sobel filter for edge detection
sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

def load_image(image_path):
    image = Image.open(image_path).convert('L')
    return np.array(image)

def save_image(image_array, output_path):
    image = Image.fromarray(image_array)
    image.save(output_path)

def edge_detection(image_section):
    grad_x = convolve(image_section, sobel_x)
    grad_y = convolve(image_section, sobel_y)
    magnitude = np.sqrt(grad_x**2 + grad_y**2)
    return np.uint8(magnitude)

def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    image_path = 'test.jpg'
    output_path = 'output_image.jpg'

    if rank == 0:
        image = load_image(image_path)
        height, width = image.shape
        sections = np.array_split(image, size, axis=0)
    else:
        sections = None

    section = comm.scatter(sections, root=0)
    processed_section = edge_detection(section)
    gathered_sections = comm.gather(processed_section, root=0)

    if rank == 0:
        result_image = np.vstack(gathered_sections)
        save_image(result_image, output_path)
        print(f"Edge detection completed. Output saved to {output_path}")

if __name__ == "__main__":
    main()