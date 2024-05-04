import mtolib.main as mto

from mtolib import attributes as atts

"""Example program - using original settings"""

# Get the input image and parameters
image, params = mto.setup()

# Pre-process the image
processed_image = mto.preprocess_image(image, params, n=2)

# Build a max tree
mt = mto.build_max_tree(processed_image, params)

#print(mt.nodes)
#print(dir(mt.nodes))
#print(mt.node_attributes[0].power, mt.node_attributes[0].volume)
print(mt.nodes[0].area, mt.node_attributes[0].volume)
print(image[400][400])

# Filter the tree and find objects
id_map, sig_ancs, mto_struct = mto.filter_tree(mt, processed_image, params)

biggest = atts.biggest_object(mt, id_map)
print(biggest)

atts.get_attributes(mto_struct, mt, biggest)
big_map = mto.biggest_map(image, id_map, biggest)

# Relabel objects for clearer visualisation
id_map = mto.relabel_segments(id_map, shuffle_labels=False)

# Generate output files
mto.generate_image(image, big_map, params)
mto.generate_parameters(image, id_map, sig_ancs, params)
