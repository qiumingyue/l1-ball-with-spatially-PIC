# l1-ball with spatially PIC
## Project description

The project aims to perform variable selection and parameter estimation for a proportional hazards (PH) model incorporating spatial effects based on partly interval-censored (PIC) data using $l_1$-ball prior. Applying Stochastic Gradient Langevin Dynamics (SGLD) principles, we have developed an efficient algorithm that can rapidly deliver results without involving complex sampling steps. The details of this algorithm can be found in the Article ```Bayesian Variable Selection with l_1-Ball for Spatially Partly Interval-Censored Data with Spatial Effects```. 


## Folder l1-geo:
The code in this folder is used to implement parameter estimation and variable selection when the spatial structure is a geo-reference.
### Dataset:
```distance_tooth.csv```: dataset of the distance matrix

### Files
```l1-geo-functions.jl``` includes all functions created from the model: spline, sampler, likelihood functions and MCMC algorithms, etc. 


```l1-geo-mainfunctions.jl``` includes the function to estimate parameter and variable selection.


```analysis.jl``` includes some hyper-parameters settings and instructions on how to run the code.

## Folder l1-lattice:
The code in this folder is used to implement parameter estimation and variable selection when the spatial structure is a lattice.
### Dataset:
```distance_tooth.csv```: dataset of the distance matrix

### Files
```l1-lattice-functions.jl``` includes all functions created from the model: spline, sampler, likelihood functions and MCMC algorithms, etc. 


```l1-lattice-mainfunctions.jl``` includes the function to estimate parameter and variable selection.


```analysis.jl``` includes some hyper-parameters settings and instructions on how to run the code.
