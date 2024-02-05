<img width="1119" alt="image" src="https://github.com/mojjaf/Self-Normalization-PET--GAN-/assets/55555705/4aff449f-91a3-4b1a-a3c2-ae8bdee51690">



DeepSN: Image-Based Self-Normalization in PET Using cGAN


Summary:
DeepSN is a GitHub project introducing an innovative image-based end-to-end self-normalization framework for positron emission tomography (PET) using conditional generative adversarial networks (cGAN). Normalization in PET corrects sensitivity non-uniformity across system lines of response (LOR), traditionally requiring a separate scan with a normalization phantom. This work proposes, for the first time, an approach that estimates normalization components directly from emission data using cGANs. The project explores different methodologies, including the use of unnormalized or geometrically corrected input data, varying input tensor shapes (2-D or 2.5-D), and employing either Pix2Pix or a polarized self-attention (PSA) Pix2Pix, a novel deep learning network developed for this work. This framework presents a promising avenue for end-to-end self-normalization in PET imaging, eliminating the need for additional normalization phantom scans.


If you use this code for your research, please cite our paper.


M. Chin, M. Jafaritadi, A. B. Franco, G. Chinn, D. Innes and C. S. Levin, "Self-Normalization for a 1-Millimeter Resolution Clinical PET System Using Deep Learning," 2022 IEEE Nuclear Science Symposium and Medical Imaging Conference (NSS/MIC), Italy, 2022, pp. 1-3, doi: 10.1109/NSS/MIC44845.2022.10398984.



@INPROCEEDINGS{10398984,
  author={Chin, Myungheon and Jafaritadi, Mojtaba and Franco, Andrew B. and Chinn, Garry and Innes, Derek and Levin, Craig S.},
  booktitle={2022 IEEE Nuclear Science Symposium and Medical Imaging Conference (NSS/MIC)}, 
  title={Self-Normalization for a 1-Millimeter Resolution Clinical PET System Using Deep Learning}, 
  year={2022},
  volume={},
  number={},
  pages={1-3},
  keywords={Deep learning;Sensitivity;Image resolution;Phantoms;Generative adversarial networks;Data models;Positron emission tomography},
  doi={10.1109/NSS/MIC44845.2022.10398984}}
