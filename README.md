# marine_snow_model

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/fpeaudecerf/marine_snow_model/HEAD)

This repository contains the code associated to the theoretical model of marine particle degradation coupled to sinking speed presented in the manuscript:

"Sinking enhances the degradation of organic particles by marine bacteria", 
Uria Alcolombri , François J. Peaudecerf , Vicente Fernandez , Lars Behrendt , Kang Soo Lee, Roman Stocker
*Nature Geosciences* (accepted 2021)  
DOI: TBD  
Link to publisher: TBD  

The files are organised as follow:
- dynamic_laws.py contains all the support functions for computing the particle dynamics and associated fluxes;
- each modelling Figure of the manuscript has its own folder FigureXX. This folder contains a main script generating the figure named FigureXX.py, the generated figure FigureXX.png and potentially ancillary files such as saved fluxes from previous computations and intermediary plots. To re-generate the figure, simply execute with Python the corresponding script;
- params.py, params_Q10.py and params_T.py contain parameter values used for computations and figure generation, and are called by the figure generating scripts;
- utils.py is a small utilities file;

- environment.yml sets the environment for Binder, so that the script can be executed in Binder without need for local Python installation. 


Access this Binder by clicking the blue badge above or at the following URL:
https://mybinder.org/v2/gh/fpeaudecerf/marine_snow_model/HEAD
