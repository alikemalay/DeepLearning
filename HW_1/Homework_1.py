import numpy as np
import scipy as sp
from scipy import ndimage, misc
import matplotlib.pyplot as plt
from PIL import Image
from PIL import ImageColor

i1 = Image.open('cat.jpg')

rgb2RED = (
    0.412453, 0.357580, 0.180423, 255,
    0.212671, 0.715160, 0.072169, 0,
    0.019334, 0.119193, 0.950227, 0 )
rgb2GREEN = (
    0.412453, 0.357580, 0.180423, 0,
    0.212671, 0.715160, 0.072169, 255,
    0.019334, 0.119193, 0.950227, 0 )
rgb2BLUE = (
    0.412453, 0.357580, 0.180423, 0,
    0.212671, 0.715160, 0.072169, 0,
    0.019334, 0.119193, 0.950227, 255 )

i2 = i1.convert("RGB", rgb2RED)
i3 = i1.convert("RGB", rgb2GREEN)
i4 = i1.convert("RGB", rgb2BLUE)

i1 = i1.resize((400,300))
i2 = i2.resize((400,300))
i3 = i3.resize((400,300))
i4 = i4.resize((400,300))

sum = Image.new('RGB',(800,600),'white')

sum.paste(i1,(0,0,400,300))

sum.paste(i3,(0,300,400,600))

sum.paste(i2,(400,0,800,300))

sum.paste(i4,(400,300,800,600))

sum.show()

sum.save('figure_2.jpg')