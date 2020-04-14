<a name="top"></a>

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/A-Breeze/French-Motor-Claims/setup?urlpath=lab)

# Analysis of French motor insurance claims
Using an open source data set of motor insurance claims for learning purposes.

<!--This table of contents is maintained *manually*-->
## Contents
1. [Important info](#Important-info)
1. [Setup](#Setup)
    - [Running the code locally](#Running-the-code-locally)
1. [Structure of the repo](#Structure-of-the-repo)
1. [Tasks: Overview](#Tasks-Overview)
1. [Tasks: Get data](#Tasks-Get-data)
    - [From Kaggle](#From-Kaggle)
1. [Tasks: Research notebooks](#Tasks-Research-notebooks)
    - [Create environment: Notebooks](#Create-environment-Notebooks)
    - [TBA](#TBA)
1. [Tasks: CI/CD](#Tasks-CICD)
    - [Run continuous integration](#Run-continuous-integration)
1. [Trouble-shooting](#Trouble-shooting)
1. [Further ideas](#Further-ideas)

<p align="right" style="text-align: right"><a href="#top">Back to top</a></p>

## Important info
- Please refer to the `LICENCE` and `IMPORTANT.md` files whenever interacting with this repo. 
- When using Binder, please note that security of the files uploaded is *not* guaranteed (as per [here](https://mybinder.readthedocs.io/en/latest/faq.html#can-i-push-data-from-my-binder-session-back-to-my-repository)).

<p align="right" style="text-align: right"><a href="#top">Back to top</a></p>

## Setup
This document describes how to run the repo using JupyterLab on Binder. To run it locally, skip this section and start from [Running the code locally](#Running-the-code-locally).

All console commands are **run from the root folder of this project** unless otherwise stated.

Start a Binder session by clicking the button at the [top](#top). Binder it uses the environment specification in `binder/` to create a conda-env called `notebook` by default. 

### Running the code locally
It *should* be possible to clone the repo and run the code in JupyterLab (or another IDE) from your own machine (i.e. not on Binder), but this hasn't been tested. Start the environment *on Windows* by creating the conda-env and starting JupyterLab as follows:
```
conda env create -f binder\environment.yml --force
conda activate french_claims_env
jupyter lab
```

<p align="right" style="text-align: right"><a href="#top">Back to top</a></p>

## Structure of the repo
The repo is loosely aiming to be structured in a similar way to the standard [Cookiecutter Data Science](https://drivendata.github.io/cookiecutter-data-science/) repo, although there are some deviations.

- `jupyter_notebooks/`: See [TBA](#TBA).
- `scripts/`: Scripts to automate the following tasks:
    - Fetching the data: : See [TBA](#TBA).
    - **TODO**: Describe additional scripts.

<p align="right" style="text-align: right"><a href="#top">Back to top</a></p>

## Tasks: Overview
Each *Tasks* section of this document has a specific package environment which is one of:
- a `venv` specified by a `requirements.txt`
- conda-env specified by an `environment.yml` (and possibly a `requirements.txt`)

In this section, we consider the `venv` option - the conda-env proceeds in an analogous way (but can take slightly longer to load). 

Each `venv` is named differently (of the form `env*`), so that you can have multiple available at one time. The workflow to setup and use each `venv` is given in the relevant *Tasks* section, and follows the pattern:
1. Setup
    ```
    conda activate notebook   # Ensure you are in the project's conda-env
    python -m venv name_of_venv   # Create venv
    source name_of_venv/bin/activate   # Activate
    # A venv comes with a specific version of pip, which may not be the latest, so we need to...
    pip install --upgrade pip   # ...upgrade pip in the venv
    pip install -r some_requirements.txt   # Install the specified package versions
    ```
1. Use the `venv` to run the code
1. (Optional) Exit and tidy up
    ```
    deactivate   # Exit from the `venv`
    rm -rf name_of_venv   # Delete it (if you want to save space, or re-build from scratch)
    ```

In many cases, the tasks listed in this document are are carried out in an automated way by the CI integration (see [Run continuous integration](#Run-continuous-integration)). **TODO**: CI tasks is not complete.

<p align="right" style="text-align: right"><a href="#top">Back to top</a></p>

## Tasks: Get data
The data is not committed to the repo. It is stored in its original location and we use scripts to automate the task of fetching it.

### From Kaggle
Data from Kaggle is fetched using the Kaggle CLI as follows:
1. \[One-off step\]: We need an API key (as per <https://www.kaggle.com/docs/api>). Sign in to Kaggle and go to: `My Account` - `API` section - `Create new API Token`. 
    - This downloads a `kaggle.json` which should normally be saved at `~/.kaggle/kaggle.json` (Linux) or `C:\Users<Windows-username>.kaggle\kaggle.json` (Windows).
1. In JupyterLab, create a kaggle.json file and move it to `.kaggle` subfolder:
    ```
    touch kaggle.json
    chmod 600 kaggle.json
    # Open the file and paste in the JSON from your `kaggle.json`
    mkdir .kaggle
    mv kaggle.json .kaggle/kaggle.json
    ```
1. Ensure the `venv` for fetching data is available and activated:
    ```
    python -m venv env_fetch_data
    source env_fetch_data/bin/activate
    pip install --upgrade pip
    pip install -r scripts/requirements.txt
    ```
1. Run the scripts to fetch the data:
    ```
    chmod +x scripts/fetch_kaggle_dataset_main.sh
    scripts/fetch_kaggle_dataset_main.sh
    chmod +x scripts/fetch_kaggle_dataset_additional.sh
    scripts/fetch_kaggle_dataset_additional.sh
    ```
1. Remove the `kaggle.json` file:
    ```
    rm -rf .kaggle
    ```

**REMEMBER** to `Expire API Token` on Kaggle (or delete the `kaggle.json` from Binder) after running (because Binder cannot be guaranteed to be secure).

<p align="right" style="text-align: right"><a href="#top">Back to top</a></p>

## Tasks: Research notebooks
Notebooks to analyse the data and iterate. Not designed to be incorporated into an automated pipeline.

#### Link to Kaggle kernels
The notebooks are designed so that the code can be run in Binder *or* on an accompanying Kaggle kernel. This allows a different method of sharing code and results, and potentially running computationally intensive commands on Kaggle only. To facilitate this: 
- The environment specification aims to match the package versions present on Kaggle.
- The notebook code is *manually* kept in sync between the repo and Kaggle kernel.

See the notes in each notebook for link to the accompanying Kaggle kernel.

### Create environment: Notebooks
Run the following to create the conda-env and register it as a kernel for use in the JupyterLab session:
```
conda env create -f jupyter_notebooks/environment.yml --force
conda activate french_claims_research_env
/srv/conda/envs/french_claims_research_env/bin/python -m ipykernel install --name french_claims_research_env --prefix /srv/conda/envs/notebook
```

The `--prefix` option ensures the new conda-env is registered as a kernel in the `notebook` conda-env (i.e. the conda-env that is running in JupyterLab).

From the JupyterLab *launcher*, you will now see there is an option to start a notebook using the new kernel (it may take a moment for this to take effect).

#### Managing Jupyter kernels
```
conda activate notebook
jupyter kernelspec list  # Get a list of the available kernels
jupyter kernelspec remove [kernel name]  # Unregister a kernel
```

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
