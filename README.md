# AMAL-Project
Implementation of the paper 'You Only Need Adversarial Supervision for Semantic Image Synthesis'

* Authors of this implementation : All authors provided the same work in terms of time, effort and contributions
  - Claire Bizon Monroc ( Discriminator, Training/test loop)
  - Amine Djeghri (Datasets, Generator)
  - Idles Mamou (scores, configs)
* `config.yml`: contains all architecture and training hyperparameters

## Training steps for ADEDataChallenge2016:
1. Clone repository
2. Download ADEChallengeData2016 dataset:
`wget http://data.csail.mit.edu/places/ADEchallenge/ADEChallengeData2016.zip`
3. Unzip `ADEChallengeData2016.zip` to replace `ADEChallengeData2016` folder
4. Adapt `config.yml` with your parameters.
5. Launch training: `python src/train.py config.yml`

## Generating Validation Data:
1. `generate.py`: generates images from the validation segmentation maps
    Generated images can then be used to compute the FID with the `pytorch-fid`
2. `scores.py`: returns the mIOU score from the validation dataset

## Other datasets: 
To use with other datasets, follow this organization:
```
data_samples:
    - name_of_dataset
        - annotations
            - training
            - validation
        - images
            - training
            - validation
```

To compute the class weights (**which are specific to your dataset**), 
use `compute_class_weights` from `utils.py`

```
cd /AMAL-Project/src
from torch.utils.data import DataLoader
from utils import compute_class_weights, get_weights
from dataset import ADEDataset # in this example we use ADE
train_dataset = ADEDataset(Path("../cityscapes_data/images/training"),Path("../cityscapes_data/annotations/training"),128)
train_loader = DataLoader(train_dataset, batch_size, True, drop_last=True)
class_weights = compute_class_weights(train_loader, C=number_of_classes, H=height_of_images, W=width_of_images) # C=51 classes, H,W of ADE20 dataset
class_weights = get_weights(class_weights, device,opt)
```
The `class_weights` can then be serialized locally to avoid being recomputed at each beginning of training.

```
torch.save(class_weights, open(local_path, "wb"))
```

In `config.yml`:
```
data:
  path: "path_to_new_dataset"
  ...
  class_weights: "path_to_serialized_weights"

# Tensorboard: path to your tensorboard directory
tb_folder: "XP"

# Checkpoints: path to your checkpoints directory
checkpoint_path: "/tempory/oasis"
```

More: 
- https://openreview.net/forum?id=yvQKLaqNE6M
- https://www.youtube.com/watch?v=3AIpPlzM_qs
