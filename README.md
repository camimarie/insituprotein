# In Situ Protein Sequencing

This repository contains the code accompanying the in situ protein sequencing manuscript. Currently, it is divided into two sections: experimental and theoretical. The experimental folder contains the image processing and figure generation code for the experimental section. The theoretical folder contains pipelines to process and analyze proteome data from mycoplasma (Species ID: 243273) and human (Species ID: 9606), as well as any other proteome of interest. It includes code for data processing, proteomic data, and scripts for simulating the sequencing process. To get started on the theoretical code, begin with the Jupyter notebooks provided in the theoretical directory.

Hard-coded variables and filenames remain. This repository is still a work in progress. Please contact Camille Mitchell at camitch@mit.edu with any questions.

## Table of Contents

- [Project Overview](#project-overview)
- [Installation](#installation)
- [Getting Started](#getting-started)
- [License](#license)
- [Acknowledgements](#acknowledgements)

## Project Overview

### Experimental

This repository contains:
- **Experimental_Plots.ipynb** - a Jupyter Notebook that generates all experimental figures.
- **Image_Processing.m** - a MATLAB file that describes how microscopy images were processed.

### Theoretical

This repository contains:
- **plots** – where plots and corresponding tables for Extended Figure 7 and Supplementary Figures are located.
- **preprocessing** – stores proteomes from UniProt.
- **sample** – where scripts and data for Part 1 resides.
- **samplereference** – where the reference fragment dataset from Part 2 resides.
- **test** – where code for Part 2 resides.
- **Introduction.ipynb** - an introductory Jupyter Notebook that provides an overview of the preprocessing steps required to run Part 1 and Part 2 of the computational sections of the paper.
- **Part1.ipynb** - a Jupyter Notebook that runs all plots (Figure 7C and Supplementary Materials) for Part 1.
- **Part2.ipynb** - a Jupyter Notebook that runs all plots (Figure 7D and Supplementary Materials) for Part 2.

## Installation

1. **Clone the repository:**

   ```bash
   git clone https://github.com/camimarie/insituprotein.git

2.  **Install dependencies:**

    Make sure you have Python installed and a new conda environment open, then run:

    ```bash
    pip install -r requirements.txt

## Getting Started

After installation, for the experimental section, we recommend opening the ```Experimental/code_for_plots_principles_insitu_protein.ipynb.ipynb``` notebook to view how the plots for the experimental section were generated.

For the theoretical section, we recommend starting with the ```Theoretical/Introduction.ipynb``` notebook. This notebook provides an introduction to the preprocessing steps, giving you a quick overview of how the in-situ protein sequencing process works. You can add your own parameters, species (other than human and mycoplasma), and run the simulations yourself. We also recommend then running the ```Theoretical/Part1.ipynb``` notebook followed by the ```Theoretical/Part2.ipynb``` notebook. Note that some of the data generation takes a significant amount of time - doing this on a cluster, or decreasing the number of iterations should help.

## License

This project is licensed under the MIT License. See the LICENSE file for details.

## Acknowledgements