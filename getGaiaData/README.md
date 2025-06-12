To calculate prior membership values:

Create a virtual envionment that has all of the necessary dependencies with:
```
conda create --name BASE9 -c conda-forge python=3.10 astropy astroquery jupyter scipy numpy matplotlib pandas pyyaml shapely bokeh=3.7.3 hdbscan os
conda activate BASE9
```
Next, download the Hunt2023.tsv, PARSEC.model.zip (then unzip), template_base9.yaml, getGaiadata.py and makePhot.ipynb into the same directory.  Also download NGC_188_GaiaData.ecsv to run the code on test data if wanted.

Run makePhot.ipynb in jupyter notebooks.
