Dual Axis Stage Post Processor
========================

This script/application is used to parse and subsequently plot data collected on the 2-axis stage.  The data is imported from a cvs file (github has sample.csv for an example) and the script finds the center of the light source, changes the origin, and subsequently generates the plots that are requested by the user.

The beginning of the script is setup to take either command line arguments defining the parameters, or you can change the parameters used in the script near the beginning by changing the variables from True to False (and vice versa) as well as changing some of the other parameters yielding the desired results.  For testing purposes, the default values should function well enough.

Any input is appreciated, especially input helping me make this code more pythonic.
