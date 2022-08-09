# Fast MRI
https://github.com/hasseledzerba/X-MRI

# Abstract

A magnetic resonance imaging (MRI) scan is conducted by acquiring lines in the k-space and full coverage of k-space is time-consuming, prompting researchers to devise new ways to reconstruct high-fidelity images based on subsampled k-space. fastMRI is a large MRI image dataset constructed by NYU Langone Health that allows researchers to train deep learning networks to reconstruct MRI images with >4-8x subsampling. Here our team aimed to build on previous self-supervised learning (SSL) approaches and construct a new convolutional neural network (CNN) with improved performance in the knee track data (1594 k-space images and 10000 DICOM images). We reported initial testing of a  SSL approach trained on a ResNet-based CNN that showed improved reconstruction than simple zero-filling and compressed sensing (CS) approaches. We also explored the effects of different hyperparameters of the SSL approach, e.g., center k-space fraction and loss mask, on training efficiencies. Lastly, we incorporated CS as an initiation approach to improve training efficiency and model accuracy. 


### TEAM MEMBERS

- Zheng Han
- Ning Yan
- Wenda Xu
- Feng Zhang 
 
