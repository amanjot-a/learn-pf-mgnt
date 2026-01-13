import numpy as np
import matplotlib.pyplot as plt

# Risk range
sigma = np.linspace(0, 0.4, 300)

# Risk aversion coefficient
A = 3

# Utility levels
U_levels = [0.03, 0.05, 0.07]

# Plot indifference curves
plt.figure(figsize=(8, 6))

for U in U_levels:
    mu = U + 0.5 * A * sigma**2
    plt.plot(sigma, mu, label=f'U = {U}')

# ----- Two Portfolios -----
# Portfolio A
sigma_A = 0.15
mu_A = 0.06

# Portfolio B
sigma_B = 0.25
mu_B = 0.075

plt.scatter(sigma_A, mu_A, color='red', s=80)
plt.scatter(sigma_B, mu_B, color='blue', s=80)

plt.annotate('Portfolio A', (sigma_A, mu_A), xytext=(sigma_A+0.01, mu_A-0.01))
plt.annotate('Portfolio B', (sigma_B, mu_B), xytext=(sigma_B+0.01, mu_B-0.01))

# Labels
plt.xlabel('Risk (Standard Deviation)')
plt.ylabel('Expected Return')
plt.title('Indifference Curves with Two Portfolios')
plt.legend()
plt.grid(True)
plt.show()
