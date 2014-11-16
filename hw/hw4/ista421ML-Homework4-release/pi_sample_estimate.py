import math
import random

def isOnCircle(x, y, r):
	if((x*x + y*y) <= r*r):
		return True
	else:
		return False	


radius = 1000000
total_points = 10000000
points_inside_circle = 0

for i in range(total_points):
	x = random.randint(-radius, radius)
	y = random.randint(-radius, radius)
	if(isOnCircle(x, y, radius)):
		points_inside_circle += 1	

pi = 4.0 * points_inside_circle / (total_points * 1.0)

print "Pi estimated is: ", pi
