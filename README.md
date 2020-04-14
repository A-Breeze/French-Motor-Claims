<a name="top"></a>

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/A-Breeze/French-Motor-Claims/setup?urlpath=lab)

# Analysis of French motor insurance claims
Using an open source data set of motor insurance claims for learning purposes.

### Important info
- Please refer to the `LICENCE` and `IMPORTANT.md` files whenever interacting with this repo. 
- When using Binder, please note that security of the files uploaded is *not* guaranteed (as per [here](https://mybinder.readthedocs.io/en/latest/faq.html#can-i-push-data-from-my-binder-session-back-to-my-repository)).

### Conventions
- All console commands are **run from the root folder of this project** unless otherwise stated.
- **TODO**: In many cases, the tasks listed in this document are are carried out in an automated way by the CI integration (see [Run continuous integration](#Run-continuous-integration)).

<!--This table of contents is maintained *manually*-->
## Contents
1. [Setup](#Setup)
    - [Running the code locally](#Running-the-code-locally)
1. [Structure of the repo](#Structure-of-the-repo)
1. [Tasks: Research notebooks](#Tasks-Research-notebooks)
    - [Create environment: Notebooks](#Create-environment-Notebooks)
    - [Get data for analysis](#Get-data-for-analysis)
    - [TBA](#TBA)
1. [Tasks: CI/CD](#Tasks-CICD)
    - [Run continuous integration](#Run-continuous-integration)
1. [Trouble-shooting](#Trouble-shooting)
1. [Further ideas](#Further-ideas)

<p align="right" style="text-align: right"><a href="#top">Back to top</a></p>

## Setup
This document describes how to run the repo using JupyterLab on Binder. To run it locally, skip this section and start from [Running the code locally](#Running-the-code-locally).

1. Start a Binder session by clicking the button at the [top](#top). Binder it uses the environment specification in `binder/` to create a conda-env called `notebook` by default. 
1. From JupyterLab, open a Console (in Linux) and run:
```
conda activate notebook
python -m venv venv
source venv/bin/activate
pip install --upgrade pip
```

### Running the code locally
It *should* be possible to clone the repo and run the code in JupyterLab (or another IDE) from your own machine (i.e. not on Binder), but this hasn't been tested. Start the environment *on Windows* by creating the conda-env and starting JupyterLab as follows:
```
conda env create -f binder\environment.yml --force
conda activate french_claims_env
jupyter lab
```

Now go back to the main [Setup](#Setup) instructions and continue from there.

<p align="right" style="text-align: right"><a href="#top">Back to top</a></p>

## Structure of the repo
The repo is loosely aiming to be structured in a similar way to the standard [Cookiecutter Data Science](https://drivendata.github.io/cookiecutter-data-science/) repo, although there are some deviations.

- `jupyter_notebooks/`: See [TBA](#TBA).
- `scripts/`: Scripts to automate the following tasks:
    - Fetching the data: : See [TBA](#TBA).
    - **TODO**: Describe additional scripts.

<p align="right" style="text-align: right"><a href="#top">Back to top</a></p>

## Tasks: Research notebooks
**TODO**: Description.

### Create environment: Notebooks
**TODO**:  Write this section

### Get data for modelling
**TODO**: Not complete

Data is required for fitting the model in the `regression_model` package. It is downloaded from Kaggle using the Kaggle CLI. For this we need an API key as per <https://www.kaggle.com/docs/api>.
- Get an API Key by signing in to Kaggle and go to: `My Account` - `API` section - `Create new API Token`. 
    - This downloads a `kaggle.json` which should normally be saved at `~/.kaggle/kaggle.json` (Linux) or `C:\Users<Windows-username>.kaggle\kaggle.json` (Windows).
- Create a `kaggle.json` file manually in JupyterLab in the project root directory (which is `~`). Then move it to a `.kaggle` folder by (in console since JupyterLab can't see folders that being with `.`):
    ```
    chmod 600 kaggle.json  # Advised to run this so it is not visible by other users
    mkdir .kaggle
    mv kaggle.json .kaggle/kaggle.json
    ```
- Now ensure the requirements for fetching data are installed and run the relevant script by:
    ```
    pip install -r ./scripts/requirements.txt
    chmod +x scripts/fetch_kaggle_dataset_main.sh
    scripts/fetch_kaggle_dataset.sh
    chmod +x scripts/fetch_kaggle_dataset_additional.sh
    scripts/fetch_kaggle_dataset.sh
    ```
- **REMEMBER** to `Expire API Token` on Kaggle (or delete the `kaggle.json` from Binder) after running (because Binder cannot be guaranteed to be secure).

### TBA

<p align="right" style="text-align: right"><a href="#top">Back to top</a></p>

## Tasks: CI/CD
### Run continuous integration
**TODO**: Write this section

### TBA

<p align="right" style="text-align: right"><a href="#top">Back to top</a></p>

## Trouble-shooting
Various notes about issues I have encountered while coding this project.

### TBA

<p align="right" style="text-align: right"><a href="#top">Back to top</a></p>

## Further ideas
Rough list of possible future ideas.
- \[None so far\]

<p align="right" style="text-align: right"><a href="#top">Back to top</a></p>
