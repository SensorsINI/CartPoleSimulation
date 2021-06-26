import imageio
import glob

frames_per_second = 1

# Name of GIF to be created
gif_name = 'my_gif'

# Path to folder with pictures from which GIF should be created
dir_path = '/Users/marcinpaluch/Desktop/Wαrm_up/f/'

paths_to_images = sorted(glob.glob(dir_path + '*.png'))
path_to_gif = dir_path + gif_name + '.gif'

# Iterate over pictures and add them to the GIF
with imageio.get_writer(path_to_gif, mode='I', fps=frames_per_second) as writer:
    for path_to_image in paths_to_images:
        image = imageio.imread(path_to_image)
        writer.append_data(image)