import numpy as np
from sklearn.preprocessing import PolynomialFeatures

x = np.random.rand(10, 150)


pol = PolynomialFeatures(3)
print(pol.fit_transform(x).shape)