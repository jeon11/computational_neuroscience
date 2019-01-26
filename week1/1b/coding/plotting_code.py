import matplotlib.pyplot as plt
import numpy as np

xvals = np.linspace(0,2*np.pi,100)
yvals = np.sin(np.linspace(0,2*np.pi,100))
plt.plot(xvals,yvals)
plt.xlabel("time") 
plt.ylabel("y values")
# this will pause execution until you close the figure
# but closing the figure also removes the plot from the workspace
#plt.show()
#print("hi!")
plt.savefig('sinplot1.pdf')