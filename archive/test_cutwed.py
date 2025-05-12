import numpy as np
from cuTWED import twed

# Create two simple time series
A = np.array([[0.0], [1.0], [2.0]], dtype=np.float64)
B = np.array([[1.0], [2.0], [3.0]], dtype=np.float64)

# Create time arrays (must be the same length as A and B)
TA = np.array([0.0, 1.0, 2.0], dtype=np.float64)
TB = np.array([0.0, 1.0, 2.0], dtype=np.float64)

# Calculate TWED distance
distance = twed(A, TA, B, TB, 1.0, 0.001, 1)
print(f"TWED distance: {distance}")
