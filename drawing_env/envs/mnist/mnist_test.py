import mnist
import numpy as np
import matplotlib.pyplot as plt

plt.figure()
images = mnist.read()
for i in range(16):
	plt.subplot(4,4,i+1)
	im = images.next()[1]
	
	fig = plt.imshow(1 - im/128, cmap='Greys')
	fig.axes.get_xaxis().set_visible(False)
	fig.axes.get_yaxis().set_visible(False)

plt.savefig('ex.png', bbox_inches='tight', pad_inches=0, dpi=300)
#fig.show()
