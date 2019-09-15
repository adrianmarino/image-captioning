# Image Captioning


## Requeriments

* [anaconda](https://www.anaconda.com/download/#linux)
* 3D video card (i.e. [GeForce GTX 1060](https://www.nvidia.com/en-us/geforce/products/10series/geforce-gtx-1060/) or higher)

## USed datasets & world embeddings

* Datasets
  * Flickr8k dataset.
  * [Flickr30k](https://www.kaggle.com/hsankesara/flickr-image-dataset/version/1#) dataset.

* Word embedings
    * [Glove word embedding](https://nlp.stanford.edu/projects/glove/)
      * [glove.6B.zip](nlp.stanford.edu/data/glove.6B.zip)
      * [glove.840B.300d](nlp.stanford.edu/data/glove.840B.300d.zip)
    * [Elmo embedding](https://allennlp.org/elmo)

## Setup

**Step 1**: Create project environment.

```bash
$ conda env create --file environment.yml
```

**Step 2**: Activate environment.
```bash
$ conda activate image-captioning
```

## Train/Test model

You can train, test and adjust model from [Image captioning notebook](https://github.com/adrianmarino/image-captioning/blob/master/image-captioning.ipynb). 
