# l1-ball with spatially PIC
## Project description

The project aims to perform variable selection and parameter estimation for a proportional hazards (PH) model incorporating spatial effects based on partly interval-censored (PIC) data using $l_1$-ball prior. Applying Stochastic Gradient Langevin Dynamics (SGLD) principles, we have developed an efficient algorithm that can rapidly deliver results without involving complex sampling steps. The details of this algorithm can be found in the Article ```Bayesian Variable Selection with l_1-Ball for Spatially Partly Interval-Censored Data with Spatial Effects```. 


## Folder:
### Dataset:
```adjacency matrix.csv```: dataset of the adjacency matrix


```simulated data--PH.csv```: simulated dataset with the true value $(0.5, 1)'$ of $\beta$:

 1.  The first two columns are the covariates
 2.  The third column is the left point of the time interval
 3.  The fourth column is the right point of the time interval
 4.  The fifth column is the censoring index
 5.  The last column is the location


### Files
```Functions.jl``` includes some functions created from the model: spline, sampler, likelihood functions and MCMC algorithms, etc. 


```analysis.jl``` includes some hyper-parameters settings and instructions on how to run the code.
