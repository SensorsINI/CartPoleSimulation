import imageio
import glob
import os

gif_name = 'my_gif'
dir_path = '/Users/marcinpaluch/Desktop/Wαrm_up/f/'
paths_to_images = sorted(glob.glob(dir_path + '*.png'))
path_to_gif = dir_path + gif_name + '.gif'
with imageio.get_writer(path_to_gif, mode='I', fps=1) as writer:
    for path_to_image in paths_to_images:
        image = imageio.imread(path_to_image)
        writer.append_data(image)
