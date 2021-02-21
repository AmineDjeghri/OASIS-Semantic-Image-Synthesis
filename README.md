# AMAL-Project
implementation of the paper 'You Only Need Adversarial Supervision for Semantic Image Synthesis'

* `config.yml`: contains all architecture and training hyperparameters

## Training steps:
1. Clone repository
2. Download ADEChallengeData2016 dataset:
`wget http://data.csail.mit.edu/places/ADEchallenge/ADEChallengeData2016.zip`
3. Unzip `ADEChallengeData2016.zip` to replace `ADEChallengeData2016` folder
4. Adapt `config.yml` with your parameters.
4. Launch training: `python src/train.py config.yml`

More: 
- https://openreview.net/forum?id=yvQKLaqNE6M
- https://www.youtube.com/watch?v=3AIpPlzM_qs