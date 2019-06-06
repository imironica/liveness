import Augmentor
p = Augmentor.Pipeline("./augument")

# Add operations to the pipeline as normal:
p.rotate(probability=0.5, max_left_rotation=5, max_right_rotation=5)
p.flip_left_right(probability=1)
p.gaussian_distortion(probability=0.7, grid_width = 3, grid_height = 3, magnitude = 2, corner = 'bell', method = 'in')
p.sample(200)
