import numpy as np
from cuTWED import twed
import inspect

# Print the function signature
print("Function signature:", inspect.signature(twed))

# Create simple test data
A = np.array([[0.0], [1.0], [2.0]], dtype=np.float64)
B = np.array([[1.0], [2.0], [3.0]], dtype=np.float64)
TA = np.array([0.0, 1.0, 2.0], dtype=np.float64)
TB = np.array([0.0, 1.0, 2.0], dtype=np.float64)

# Try with different parameter combinations
try:
    print("\nTrying positional arguments:")
    distance = twed(A, TA, B, TB, 1.0, 0.01, 2)
    print(f"Result: {distance}")
except Exception as e:
    print(f"Error with positional args: {e}")

# Try to extract the actual documentation
print("\nDocumentation:")
print(twed.__doc__)
