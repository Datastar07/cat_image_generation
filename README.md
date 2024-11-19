# cat_image_generation

## Used Data(Cat)

- Training image - 9000


# Two types of model,
1) Noise schedular:
    - the noise scheduler works by adding a random noise to the image based on the time step
2) U-net:
    - the U-Net consists of 5 down and 5 up layers with 10 dense layers for time embeddings that are added in each layer to the output