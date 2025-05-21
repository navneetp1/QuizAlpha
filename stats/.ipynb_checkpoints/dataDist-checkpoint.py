import matplotlib.pyplot as plt
import numpy as np

keys = np.load('data/keys.npy')

data = {}
for key in keys:
    data[key] = data.get(key, 0) + 1

data = sorted(data.items(), key = lambda x: x[1], reverse=True)

x = [d[0] for d in data]
y = [d[1] for d in data]

print("No of classes:", len(data))
print("No of data points:", len(keys))

plt.bar(x,y)
plt.plot()

plt.show()