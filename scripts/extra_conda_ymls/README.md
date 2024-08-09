# Notes on packages used in conda

- In general, the "conda_just_packages.yml" may be useful for seeing which packages were used.

- The "conda_linux_Borah.yml" can be used to build on linux much faster with all of the dependencies already solved.

- There were issues with numpy>=2.0 working with Cython code. And so this is fixed currently at 1.26.4.

- There were issues running mpi4py==4.0.0 on Borah. This is fixed at v3.1.4.