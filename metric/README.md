# Metrics

## Intro

### FID

FID is a measure of similarity between features of two datasets. It was shown to correlate well with human judgement of visual quality and is most often used to evaluate the quality of samples of Generative Adversarial Networks. 

- Implementation: https://github.com/mseitzer/pytorch-fid

### MMD

MMD measures the distance between two distributions. We can calculate the MMD between the generated dataset and the real dataset. 

### L1 RECONSTRUCTION





## Code

Modify the path to real dataset and generated dataset.

Specify 'L1' or 'MMD'.

Specify 'binary'. For RBM, since the generated images are binary, we set binary as True. For GAN, we may set it as False.

Corresponding images (a pair of real one and generated one) should have the same names but in different directory.

Lastly, just run the code.