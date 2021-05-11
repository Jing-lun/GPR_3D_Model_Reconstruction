3D Subsurface Object Reconstruction Model basedon Ground Penetration Radar
====
This is an end-to-end model that efficient learns the 3D shape of subsurface objects from GPR 2D data. The reconstructed map is represented in occupancy volumetric format.

We also create a concrete slab dataset for DNN-based GPR inspection and reconstruction purpose. Each concrete slab not only contains cylinder objects, but also include the sphere and cubic objects, with different size, location and orientation.

If you're interested at our work, please cite the following papers:
```
@inproceedings{feng2021subsurface,
title={Subsurface Pipes Detection Using DNN-based Back Projection on GPR Data},
author={Feng, Jinglun and Yang, Liang and Wang, Haiyan and Tian, Yingli and Xiao, Jizhong},
booktitle={Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision},
pages={266--275},
year={2021}
}

@article{feng2020gpr,
title={GPR-based Model Reconstruction System for Underground Utilities Using GPRNet},
author={Feng, Jinglun and Yang, Liang and Hoxha, Ejup and Sotnikov, Stanislav and Sanakov, Diar and Xiao, Jizhong},
journal={arXiv preprint arXiv:2011.02635},
year={2020}
}
```

Overview
----
Ground Penetration Radar (GPR) is a well-known non-destructively testing (NDT) tool in infrastructure inspection and is widely used to locate and map the subsurface targets. Nowadays, understanding the subsurface 3D world gains more attention and discussion in the GPR-related field. However, an intuitive 3D reconstruction representation of the underground objects is still an open problem since 2D GPR data representation and interpretation don't follow the perspective transformation. In this work, we investigate GPR-based 3D subsurface target reconstruction from a back-projection perspective. We formulate this reconstruction procedure as an implicit back-projection from 2D to 3D representations and propose an end-to-end network to implement this conversion. Our end-to-end model doesn't require any pre-processing on the GPR data compared with conventional approaches, and the network is also able to generate a 3D volumetric map through the GPR data. Results show superior performance and better perception ability for subsurface 3D object reconstruction when compared with other methods.

# Installation
You can install them by:
```
conda install -y pytorch=1.5.0 torchvision=0.6.0 cudatoolkit=10.2 -c pytorch
pip install \
	open3d==0.9.0.0 \
	scipy == 1.5.4\
	h5py == 3.1.0\
```

# Dataset
Our [dataset](https://www.dropbox.com/sh/q5sfb7ciys3v2mr/AACHhj_FOLiNBRm5XvorIpd2a?dl=0) contains 627 different slab models cretated by gprMax. The surrounding dielectric of each slab model is set as `half_space` while the wavefront is `gaussiandotnorm`.

# Test
You can have a result of test run using
```
python test.py --model 'check_points/CP_epoch101.pth' --input' 'PATH TO INPUT' --gt 'PATH TO GROUND TRUTH' --file 'PATH TO TEST FILE'
```

# Train
In order to train our dataset, you need to download our dataset shown above.

Than you can train it using `train.py`

Configuration is shown in `train.py` code. We also provide the default value for all the configuration settings.

# License

License for source code corresponding to:

Copyright (c) 2021 Jinglun Feng

Please read carefully the following terms and conditions and any accompanying documentation before you download and/or use this software and associated documentation files (the "Software").

The authors hereby grant you a non-exclusive, non-transferable, free of charge right to copy, modify, merge, publish, distribute, and sublicense the Software for the sole purpose of performing non-commercial scientific research, non-commercial education, or non-commercial artistic projects.

Any other use, in particular any use for commercial purposes, is prohibited. This includes, without limitation, incorporation in a commercial product, use in a commercial service, or production of other artefacts for commercial purposes.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

You understand and agree that the authors are under no obligation to provide either maintenance services, update services, notices of latent defects, or corrections of defects with regard to the Software. The authors nevertheless reserve the right to update, modify, or discontinue the Software at any time.

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software. You agree to cite the corresponding papers (see above) in documents and papers that report on research using the Software.
