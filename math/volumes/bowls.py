import math
import sympy as sp

# Important measurements
#TODO: We will have to pull these from photos, figure out how to use the hand as a reference in the images and extract the length, IMO its simply a "rapport" between pixels.
#TODO: Your hand if we know your true hand length and width we can use it to find how much centimeters a pixel is.
radius = 3.3
total_length = 14.1
max_height = 8

def paraboloid_plates(radius, total_length, max_height):
    # Find the cylinder volume
    cylinder_volume = max_height * (radius ** 2) * math.pi

    #Find volume of outer edge
    outer_edge = (total_length / 2) - radius
    increase_rate = (max_height) / ((outer_edge) ** 2)
    r = sp.Symbol('r')
    z = sp.Symbol('z')
    t = sp.Symbol('t')
    upper = increase_rate * ((r - radius) ** 2)
    integral = sp.Integral((r), (z, 0, upper), (r, radius, radius + outer_edge), (t, 0, 2 * math.pi))
    outer_volume = integral.doit()
    
    return cylinder_volume + outer_volume

print(paraboloid_plates(3.3, 14.1, 8))

    

