import imageio
filenames = glob.glob('/Users/marcinpaluch/Desktop/Wαrm_up/f/' + '*.csv')
with imageio.get_writer('/path/to/movie.gif', mode='I') as writer:
    for filename in filenames:
        image = imageio.imread(filename)
        writer.append_data(image)