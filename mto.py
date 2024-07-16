import mtolib.main as mto

import numpy as np

from mtolib import attributes as atts
from skimage import restoration

"""Example program - using original settings"""

# Get the input image and parameters
image, psf, params = mto.setup()

# Pre-process the image
processed_image = mto.preprocess_image(image, params, n=2)

# Build a max tree
mt = mto.build_max_tree(processed_image, params)

# Filter the tree and find objects
id_map, sig_ancs, mto_struct = mto.filter_tree(mt, processed_image, params)

biggest = atts.biggest_object(mt, id_map)
print(biggest)

atts.get_attributes(mto_struct, mt, biggest, psf=psf)
big_map = mto.biggest_map(image, id_map, biggest)

# Relabel objects for clearer visualisation
id_map = mto.relabel_segments(id_map, shuffle_labels=False)

# Generate output files
mto.generate_image(image, big_map, params)
mto.generate_parameters(image, id_map, sig_ancs, params)
