# Image Captioning


## Requeriments

* [anaconda/anaconda-natigator](https://www.anaconda.com/download/#linux)
* 3D video card (i.e. [GeForce GTX 1060](https://www.nvidia.com/en-us/geforce/products/10series/geforce-gtx-1060/) or higher)

## Used datasets & world embeddings

* Datasets
  * Flickr8k dataset.
  * [Flickr30k](https://www.kaggle.com/hsankesara/flickr-image-dataset/version/1#) dataset.

* Word embedings
    * [Glove word embedding](https://nlp.stanford.edu/projects/glove/)
      * [glove.6B.zip](nlp.stanford.edu/data/glove.6B.zip)
      * [glove.840B.300d](nlp.stanford.edu/data/glove.840B.300d.zip)
    * [Elmo embedding](https://allennlp.org/elmo)

## Setup environment

**Step 1**: Create project environment.

```bash
$ conda env create --file environment.yml
```

**Step 2**: Activate environment.
```bash
$ conda activate image-captioning
```

## Prepare dataset

**Step 1**: Download & prepare word embeddings to use:
```bash
$ ./setup_word_embedings
```

**Step 2**: Download Flickr8K dataset .torrent from [this](http://academictorrents.com/details/9dea07ba660a722ae1008c4c8afdd303b6f6e53b).

**Step 3**: Once downloaded the Flickr directory, make *dataset* directory under project path and copy downloaded directory to this.

```bash
$ mkdir -p dataset
$ cp -rf ~/Downloads/Flickr8k ~/project/dataset/flickr8k
```

**Step 4**: Login an download [Flickr30k](https://www.kaggle.com/hsankesara/flickr-image-dataset/version/1#) dataset from *kaggle*.

**Step 5**: Copy downloaded file to project/dataset path and unzip this.

```bash
$ cp -rf ~/Downloads/flickr30k_images.zip ~/project/dataset/
$ cd ~/project/dataset
$ unzip flickr30k_images.zip
```

**Step 6**: finally change directories structure like this.

<img alt="dataset structure" src="https://raw.githubusercontent.com/adrianmarino/image-captioning/master/images/dataset-tree.png" height="250" />


## Train/Test model

You can train, test and adjust model from [Image captioning notebook](https://github.com/adrianmarino/image-captioning/blob/master/image-captioning.ipynb). 
