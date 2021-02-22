# AMAL-Project
implementation of the paper 'You Only Need Adversarial Supervision for Semantic Image Synthesis'

* `config.yml`: contains all architecture and training hyperparameters

## Training steps for ADEDataChallenge2016:
1. Clone repository
2. Download ADEChallengeData2016 dataset:
`wget http://data.csail.mit.edu/places/ADEchallenge/ADEChallengeData2016.zip`
3. Unzip `ADEChallengeData2016.zip` to replace `ADEChallengeData2016` folder
4. Adapt `config.yml` with your parameters.
4. Launch training: `python src/train.py config.yml`

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
use `compute_class_weights` from `utils.py`. 

```
train_loader = DataLoader(train_dataset, batch_size, True, drop_last=True)
class_weights = compute_class_weights(train_loader, C=number_of_classes, H=height_of_images, W=width_of_images)
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
