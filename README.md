# Global Optical Snow properties via High-speed Algorithm With K-means clustering (GOSHAWK)
[![DOI](https://zenodo.org/badge/756454442.svg)](https://zenodo.org/doi/10.5281/zenodo.10652709)

<p align="center">
    <img src="docs/goshawk.png" alt="goshawk" width="300"/>
</p>

This repository was created for use in my PhD dissertation and will be updated to the best I can moving forward. Pull requests, contributions, and issues are welcome!

GOSHAWK can be used to take in imaging spectroscopy radiance data (microW cm-2 nm-1 sr-1) and output optical snow properties. Our algorithm solves for these properties simultaneously using numerical optimization of surface and atmospheric state variables. This code expects the light wavelengths to range from roughly 350-2500nm. 

In it's current state, GOSHAWK expects PRISMA L1A processed via [SISTER](https://github.com/EnSpec/sister) or EMIT L1B processed via SISTER. Full disclosure the EMIT implementation has not been formally validated , see our paper on testing PRISMA. But the code is structured to take the EMIT data. Other missions may be adapted using this framework.

The code was designed to be ran on any kind of Linux clusters mananged by SLURM (e.g., Boise State's **R2** and **Borah** clusters, https://bsu-docs.readthedocs.io/en/latest/) with large amounts of CPU across many nodes. It is __not recommended__ running this on a personal machine, unless your personal machine has access to approx. 192 CPU and 192 GB RAM.  The biggest bottleneck for this being accessible to average machine in 2024 (n-cpu ~ 4-10) is __libRadtran__ LUT for each image. The other major bottleneck is the numerical optimizations themselves. Even with dimensionality reduction the average number of optimizations is around 50-75k for a PRISMA-like image. With each optimization taking a few seconds, these tasks add up significantly on personal machines. SLSQP is a quality constrained,non-linear solver and is quite fast already written in fortran. Therefore, there is not a really obvious solution to speed this up for personal machine use ... __TLDR; use this code on a Linux cluster.__


## 1. Required image inputs :framed_picture:

- Before runing GOSHAWK, you will need to process PRISMA L1A through SISTER
- Required file structure:
    - PRS_20210123184757_20210123184801_0001_rdn_prj
    - PRS_20210123184757_20210123184801_0001_rdn_prj.hdr
    - PRS_20210123184757_20210123184801_0001_loc_prj
    - PRS_20210123184757_20210123184801_0001_loc_prj.hdr
    - PRS_20210123184757_20210123184801_0001_obs_prj
    - PRS_20210123184757_20210123184801_0001_obs_prj.hdr


## 2. Download GOSHAWK :inbox_tray:

- Download the code via the releases tab (V1.0.0)
- Or if you are interested in contributing you can `git clone` the most recent code


## 3. Set up conda environment via miniconda :snake:
- If you are using Boise State cluster you can not create conda env on head node in terminal.. So there is a script `goshawk-conda.bash` you can submit to create one. Note that all of the bash scripts have my personal paths in them and would need to be changed.
- Once created, run the following command prior to running the scripts (you always need to activate before running).

```
$ conda activate goshawk
```


## 4. Enable Google Earth Engine :earth_asia:
- Create a Google Earth Engine Service account: https://developers.google.com/earth-engine/guides/service_account 
- Enable Google Earth Engine API
- Create a private key
- Save .json private key file to GOSHAWK directory

NOTE: Google's website above claims that a service account is not needed. However, this was the only way I could get Earth Engine to work when ran on Linux cluster. I can keep testing with other clusters to see if this is an isolated issue, but for now I will keep this as the method.


## 5. Download spectral library :books:
download __ecospeclib-all__ from ECOSTRESS and save unzipped folder to GOSHAWK directory. To help keep this code accessible, I have copied all of the files actually used here. But please note there are many many more spectra available in this library.
- folder should be named `ecospeclib-all`
- https://speclib.jpl.nasa.gov/ 



## 6. Download libRadtran :sunny:
The following instructions can be used to setup on cluster:

```
# Run each step in command line in the GOSHAWK directory:
#        wget -nv http://www.libradtran.org/download/libRadtran-2.0.4.tar.gz
#        tar -xf libRadtran-2.0.4.tar.gz
#        cd libRadtran-2.0.4
#        wget -nv http://www.meteo.physik.uni-muenchen.de/~libradtran/lib/exe/fetch.php?media=download:reptran_2017_all.tar.gz -O reptran-2017-all.tar.gz
#        tar -xf reptran-2017-all.tar.gz
#        wget -nv http://www.meteo.physik.uni-muenchen.de/~libradtran/lib/exe/fetch.php?media=download:optprop_v2.1.tar.gz -O optprop_v2.1.tar.gz
#        tar -xf optprop_v2.1.tar.gz
```

- Please see `goshawk-pt1.bash` that compiles and runs it. NOTE: if your HPC lib are different -or changed- this may cause things to break.. Most likely you will just need to find the correct module to load by searching the available cluster modules.
- Also note, that libRadtran compiles with python2, and so your cluster will need to have that downloaded (see `goshawk-pt1.bash` )


## 7. Run the model :rocket:

- The image can be ran with bash scripts (sbatch pt 1-3..)
- Note: these 3 scripts are where the user changes inputs. 
- each one can be ran by running the following in the terminal, `sbatch goshawk-pt1.bash`


Please note that variable `path_to_img_base` should look like the following (please use full path):
- '/home/brentwilder/PRS_20221208184337_20221208184341_0001'

## References
- Wilder, Brenton A., et al. "Computationally efficient retrieval of snow surface properties from spaceborne imaging spectroscopy measurements through dimensionality reduction using k-means spectral clustering." IEEE Journal of Selected Topics in Applied Earth Observations and Remote Sensing (2024).

## Ideas for improvements
- better clouds filtering algorithm within goshawk...


## Current speeds (on PRISMA PRS_L1_STD_OFFL_20210429180418_20210429180422_0001)
- PT1: 48 CPU - Run time 00:14:05
- PT2: 192 CPU - Run time 00:09:34
- PT3: 48 CPU - Run time 00:01:54