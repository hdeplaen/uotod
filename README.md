# Unbalanced Optimal Transport: A Unified Framework for Object Detection
<a href="https://hdeplaen.github.io/uotod/" target="_blank">Presentation</a>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
<a href="https://openaccess.thecvf.com/content/CVPR2023/papers/De_Plaen_Unbalanced_Optimal_Transport_A_Unified_Framework_for_Object_Detection_CVPR_2023_paper.pdf" target="_blank">Paper</a>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
<a href="https://openaccess.thecvf.com/content/CVPR2023/supplemental/De_Plaen_Unbalanced_Optimal_Transport_CVPR_2023_supplemental.pdf" target="_blank">Supplementary</a>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
<a href="https://uotod.readthedocs.io/en/latest/" target="_blank">Documentation</a>&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;


![GitHub License](https://img.shields.io/github/license/hdeplaen/uotod)
![PyPI - Downloads](https://img.shields.io/pypi/dm/uotod)
![PyPI - Version](https://img.shields.io/pypi/v/uotod)
[![Documentation Status](https://readthedocs.org/projects/uotod/badge/?version=latest)](https://uotod.readthedocs.io/en/latest/?badge=latest)

[//]: # ([![Test Status]&#40;https://github.com/hdeplaen/uotod/actions/workflows/test.yaml/badge.svg?branch=main&#41;]&#40;https://github.com/hdeplaen/uotod/actions/workflows/test.yaml&#41;)

[//]: # ([![Build Status]&#40;https://github.com/hdeplaen/uotod/actions/workflows/build.yaml/badge.svg?branch=main&#41;]&#40;https://github.com/hdeplaen/uotod/actions/workflows/build.yaml&#41;)

[//]: # (![GitHub all releases]&#40;https://img.shields.io/github/downloads/hdeplaen/uotod/total&#41;)

H. De Plaen, P.-F. De Plaen, J. A. K. Suykens, M. Proesmans, T. Tuytelaars, and L. Van Gool, “Unbalanced Optimal Transport: A Unified Framework for Object Detection,” in *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*, Jun. 2023, pp. 3198–3207.

This work has be presented at CVPR 2023 in Vancouver, Canada. The paper and additional resources can be found on the [following website](https://hdeplaen.github.io/uotod/). The paper is in open access and can also be found on the [CVF website](https://openaccess.thecvf.com/content/CVPR2023/html/De_Plaen_Unbalanced_Optimal_Transport_A_Unified_Framework_for_Object_Detection_CVPR_2023_paper.html) as well as on [IEEE Xplore](https://ieeexplore.ieee.org/document/10204500).

![Different matching strategies. All are particular cases of Unbalanced Optimal Transport](img/illustration.png)

## Abstract
*TL;DR: We introduce a much more versatile new class of matching strategies unifying many existing ones, as well as being well suited for GPUs.*

During training, supervised object detection tries to correctly match the predicted bounding boxes and associated classification scores to the ground truth. This is essential to determine which predictions are to be pushed towards which solutions, or to be discarded. Popular matching strategies include matching to the closest ground truth box (mostly used in combination with anchors), or matching via the Hungarian algorithm (mostly used in anchor-free methods). Each of these strategies comes with its own properties, underlying losses, and heuristics. We show how Unbalanced Optimal Transport unifies these different approaches and opens a whole continuum of methods in between. This allows for a finer selection of the desired properties. Experimentally, we show that training an object detection model with Unbalanced Optimal Transport is able to reach the state-of-the-art both in terms of Average Precision and Average Recall as well as to provide a faster initial convergence. The approach is well suited for GPU implementation, which proves to be an advantage for large-scale models.

## Install
### PyPI

Using PyPI, it suffices to run `pip install uotod`. Just rerun this command to update the package to its newest version.

### Build From Source

You can also download it directly from the GitHub repository, then build and install it.

```bash
git clone --recursive https://github.com/hdeplaen/uotod
cd uotod
python3 -m pip install -r requirements.txt
python3 -m setup build
python3 -m pip install
 ```

### Compiled Acceleration

The package is **available on all dsitributions** and runs well. However, only the combinations marked with a green ✅ can 
take advantage of the compiled version of Sinkhorn's algorithm directly from PyPI. On a not support combination, you may always build it 
from the source to also have access to Sinkhorn's compiled version of the algorithm. Nevertheless, the PyTorch implementation 
of **Sinkhorn's algorithm is always available** (used by default), this only refers to an additional compiled version. 

| **OS**          	| **Linux** 	| **MacOS** 	 | **Windows** 	|
|-----------------	|:---------:	|:-----------:|:-----------:	|
| **Python 3.8**  	|     ✅     	|   ✅     	   |      ☑️      	|
| **Python 3.9**  	|     ✅     	|   ✅     	   |      ☑️      	|
| **Python 3.10** 	|     ✅     	|   ✅     	   |      ☑️      	|
| **Python 3.11** 	|     ✅     	|   ✅     	   |      ☑️      	|
| **Python 3.12** 	|     ✅     	|     ☑️       |      ☑️      	|

- ✅: Python implementation + compiled acceleration, _both directly from PyPI_
- ☑️: Python implementation _directly from PyPI_ (+ possible compiled acceleration if building from source)

## Examples

### OT matching with GIoU loss:

```python
from uotod.match import BalancedSinkhorn
from uotod.loss import GIoULoss

ot = BalancedSinkhorn(
    loc_match_module=GIoULoss(reduction="none"),
    background_cost=0.,
)
```

### Hungarian matching (bipartite) with GIoU loss:

```python
from uotod.match import Hungarian
from uotod.loss import GIoULoss

hungarian = Hungarian(
    loc_match_module=GIoULoss(reduction="none"),
    background_cost=0.,
)
```

### Loss from SSD solved with Unbalanced Optimal Transport:

```python
from torch.nn import L1Loss, CrossEntropyLoss

from uotod.match import UnbalancedSinkhorn
from uotod.loss import DetectionLoss, IoULoss

matching_method = UnbalancedSinkhorn(
    cls_match_module=None,  # No classification cost
    loc_match_module=IoULoss(reduction="none"),
    background_cost=0.5,  # Threshold for matching to background
    is_anchor_based=True,  # Use anchor-based matching
    reg_target=1e-3,  # Relax the constraint that each ground truth is matched to exactly one prediction
)

criterion = DetectionLoss(
    matching_method=matching_method,
    cls_loss_module=CrossEntropyLoss(reduction="none"),
    loc_loss_module=L1Loss(reduction="none"),
)

preds = ...
targets = ...
anchors = ...

loss_value = criterion(preds, targets, anchors)
```

### Loss from DETR solved with Optimal Transport (with 2 classes):

```python
import torch
from torch.nn import L1Loss, CrossEntropyLoss

from uotod.match import BalancedSinkhorn
from uotod.loss import DetectionLoss
from uotod.loss import MultipleObjectiveLoss, GIoULoss, NegativeProbLoss

matching_method = BalancedSinkhorn(
    cls_match_module=NegativeProbLoss(reduction="none"),
    loc_match_module=MultipleObjectiveLoss(
        losses=[GIoULoss(reduction="none"), L1Loss(reduction="none")],
        weights=[1., 5.],
    ),
    background_cost=0.,  # Does not influence the matching when using balanced OT
)

criterion = DetectionLoss(
    matching_method=matching_method,
    cls_loss_module=CrossEntropyLoss(
        reduction="none",
        weight=torch.tensor([0.1, 1., 1.])  # down-weight the loss for the no-object class
    ),
    loc_loss_module=MultipleObjectiveLoss(
        losses=[GIoULoss(reduction="none"), L1Loss(reduction="none")],
        weights=[1., 5.],
    ),
)

preds = ...
targets = ...
loss_value = criterion(preds, targets)
```


## Color Boxes Dataset

The Color Boxes dataset is composed of 4800 training images and 960 validation images. The images are 500x400 pixels and contain 0 to 30 colored boxes. The boxes are randomly placed and have random colors, which defines the class of the box (20 classes). The dataset is designed to be simple and easy to use for testing object detection algorithms.

You can download the dataset at the following address: [Color Boxes Dataset](https://homes.esat.kuleuven.be/~pdeplaen/colorboxes.zip). 

The dataset uses the COCO annotation format and is organized as follows:
```
path/to/colorboxes/
  annotations/ 
    instances_train.json
    instances_val.json
  train/
    00000.jpg
    00001.jpg
    ...
  val/
    00000.jpg
    00001.jpg
    ...
```



![Examples from the Color Boxes Dataset](img/colorboxes.png)

## Citation
If you make any use of our work, please refer to us as:
```bibtex
@InProceedings{De_Plaen_2023_CVPR,
    author    = {De Plaen, Henri and De Plaen, Pierre-Fran\c{c}ois and Suykens, Johan A. K. and Proesmans, Marc and Tuytelaars, Tinne and Van Gool, Luc},
    title     = {Unbalanced Optimal Transport: A Unified Framework for Object Detection},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2023},
    pages     = {3198-3207}
}
```

## Acknowledgements
EU: The research leading to these results has received funding from the European Research Council under the European Union’s Horizon 2020 research and innovation program / ERC Advanced Grant E-DUALITY (787960). This paper reflects only the authors’ views and the Union is not liable for any use that may be made of the contained information. Research Council KUL: Optimization frameworks for deep kernel machines C14/18/068. Flemish Government: FWO: projects: GOA4917N (Deep Restricted Kernel Machines: Methods and Foundations), PhD/Postdoc grant; This research received funding from the Flemish Government (AI Research Program). All the authors are also affiliated to Leuven.AI - KU Leuven institute for AI, B-3000, Leuven, Belgium.
<p style="text-align: center;">
<img src="https://hdeplaen.github.io/uotod/img/eu.png" alt="European Union" style="height:80px;"/>
<img src="https://hdeplaen.github.io/uotod/img/erc.png" alt="European Research Council" style="height:80px;"/>
<img src="https://hdeplaen.github.io/uotod/img/fwo.png" alt="Fonds voor Wetenschappelijk Onderzoek" style="height:80px;"/>
<img src="https://hdeplaen.github.io/uotod/img/vl.png" alt="Flanders AI" style="height:80px;"/>
<img src="https://hdeplaen.github.io/uotod/img/kuleuven.png" alt="KU Leuven" style="height:80px;"/>
</p>