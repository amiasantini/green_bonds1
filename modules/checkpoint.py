#Ellipse checkpoint attempt
import math
 
# Function to check the point
def checkpoint( x, y, e, lambda_):
 
    # checking the equation of
    # ellipse with the given point
    p = ((math.pow((e[0][0]*x+e[1][0]*y), 2) // math.pow(lambda_[0][0], 2)) +
         (math.pow(e[0][1]*x+e[1][1]*y, 2) // math.pow(lambda_[1][1], 2)))
  
    if (p > 1):
        return False
 
    elif (p == 1):
        return True
 
    else:
        return True