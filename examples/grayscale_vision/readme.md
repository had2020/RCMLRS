# Point
This is a very simple ML network, to showcase this framework,
it uses the blackwhite dataset for training .
Its goal is to decide if an image is more black or more white.

# How
Input 50x50 png, use must name it in the arguments to the scan_image() method.
There is a traning method and a usage mode, off a saved model.
The model will give off a proablity 1 being black and 0 being white.
It trains of the 0,0,0 for black and 255,255,255 for white in RGB, to decide.
