# Ice layer formation
## Authors
- Mohammad Afzal Shadab<sup>1,2,3</sup> (mashadab@utexas.edu)
- Surendra Adhikari<sup>2</sup>
- Anja Rutishauser<sup>4,5</sup>
- Cyril Grima<sup>3,4</sup>
- Marc Andre Hesse<sup>1,3,4</sup> (mhesse@jsg.utexas.edu)

## Affiliations
1 Oden Institute for Computational Engineering and Sciences, The University of Texas at Austin

2 NASA Jet Propulsion Laboratory, California Institute of Technology

3 University of Texas Institute for Geophysics   

4 Jackson School of Geosciences, The University of Texas at Austin

5 Department of Glaciology and Climate, Geological Survey of Denmark and Greenland


## Citation
[1] Shadab, Adhikari, Rutishauser, Grima, and Hesse, 202X. Melt supply variability controls the formation of ice layers in Greenland firn (under review).
[2] Shadab, Adhikari, Rutishauser, Grima, and Hesse, 2023. Mechanism and Factors Controlling Ice Layer Formation in Glacial Firn. AGU Fall Meeting C43D-1632. 

## Getting Started

### Dependences

The codes require the following packages to function:
- [Python](https://www.python.org/) version 3.5+
- [Numpy](http://www.numpy.org/) >= 1.16
- [scipy](https://www.scipy.org/) >=1.5
- [matplotlib](https://matplotlib.org/) >=3.3.4

Tested on
- [Python](https://www.python.org/) version 3.9.14
- [Numpy](http://www.numpy.org/) version 1.25.2
- [scipy](https://www.scipy.org/) version 1.11.2
- [matplotlib](https://matplotlib.org/) version 3.7.2


### Quick Usage
After cloning the repository and installing the required libraries, run the Python files corresponding to the figure numbers as given in the paper. Codes can be run either directly or on an IDE such as Anaconda Spyder. `Solver` is the folder containing the auxiliaries.
Comments are provided in the code. All codes to run are provided in `Main` folder. Output figures are located in the Figures folder.

#### Figures plotted by corresponding Python file in `Main/Codes/` folder with rough times on a Modern 4 core PC
Figure 2: (a to d) figure2atod_singlecase.py   <Approx runtime: 5 mins >
          (e to f) figure2ef_regime_diagram.py <Approx runtime: 1-5 seconds >
          
Figure 3: figre3_multiple_cycles_newBC.py  <Approx runtime: 10 mins >

Figure 4 (complete figure, requires field data from within the folder): figure4_composition_new_with_crater_surface_tracking.py <Approx runtime: 20 mins >


<p align="center">
<img src="./Main/Figures/validation_figure.png" height="700">
</p>
Figure : Infiltration into a multilayered firn with porosity and temperature decay with depth: (a) Schematic diagram showing all of the layers. The resulting evolution of firn (b) porosity φ, (c) liquid water content LWC or volume fraction of water and (d) temperature T evaluated by the numerical simulation in the absence of heat diffusion. Here all dashed lines show analytic solutions computed from the unified kinematic wave theory proposed by Shadab et al., Arxiv (2024) [1]. The thin, grey dashed lines show theoretically calculated dimensionless times of saturation and ponding. The theoretical evolutions of the initial wetting front is shown with red dashed line  and the dynamics of saturated region after wetting front reaches z = 5 m is shown by blue and green dashed lines.


## References:
1. Shadab, M.A., Rutishauser, A., Grima, C. and Hesse, M.A., 2024. A Unified Kinematic Wave Theory for Melt Infiltration into Firn. ArXiv. 
2. Shadab, M.A., Rutishauser, A., Grima, C. and Hesse, M.A., 2023. A Unified Kinematic Wave Theory for Melt Infiltration into Firn. AGU23.
