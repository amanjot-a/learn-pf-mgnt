import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import skew

# Realistic sample data (e.g., income-like data)
data = np.array([
    22, 25, 27, 30, 32, 35, 38, 40, 42, 45,
    48, 50, 55, 60, 65, 70, 80, 90, 120, 200
])

# Calculate skewness
data_skewness = skew(data)

# Plot histogram
plt.figure()
plt.hist(data, bins=8)
plt.axvline(np.mean(data), linestyle='dashed', linewidth=1)
plt.axvline(np.median(data), linestyle='dotted', linewidth=1)

plt.title(f"Real Data Skewness (Skew = {data_skewness:.2f})")
plt.xlabel("Value")
plt.ylabel("Frequency")
plt.show()

print("Skewness:", data_skewness)
