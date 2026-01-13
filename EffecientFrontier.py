import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# 1. Inputs (CFA-style)
# -----------------------------
rf = 0.05                 # Risk-free rate (5%)
mu1, mu2 = 0.12, 0.18     # Expected returns
s1, s2 = 0.15, 0.22       # Standard deviations
rho = 0.3                # Correlation

# -----------------------------
# 2. Efficient Frontier (2 risky assets)
# -----------------------------
w = np.linspace(0, 1, 100)

# Portfolio return
rp = w * mu1 + (1 - w) * mu2

# Portfolio risk
sp = np.sqrt(
    (w**2) * s1**2 +
    ((1 - w)**2) * s2**2 +
    2 * w * (1 - w) * rho * s1 * s2
)

# -----------------------------
# 3. Tangency Portfolio (given in your diagram)
# -----------------------------
mu_tan = 0.18     # 18%
sigma_tan = 0.22  # 22%

# CAL slope (Sharpe ratio)
cal_slope = (mu_tan - rf) / sigma_tan

# CAL line
sigma_cal = np.linspace(0, 0.35, 100)
return_cal = rf + cal_slope * sigma_cal

# -----------------------------
# 4. Plot
# -----------------------------
plt.figure(figsize=(10, 6))

# Efficient frontier
plt.plot(sp, rp, label="Efficient Frontier", linewidth=2)

# Capital Allocation Line
plt.plot(sigma_cal, return_cal, label="Capital Allocation Line (CAL)", linestyle="--")

# Risk-free asset
plt.scatter(0, rf, color="red", zorder=5)
plt.text(0.002, rf, "Risk-free asset (Rf)", fontsize=9)

# Tangency portfolio
plt.scatter(sigma_tan, mu_tan, color="black", zorder=5)
plt.text(sigma_tan + 0.005, mu_tan, "Tangency Portfolio", fontsize=9)

# Labels
plt.xlabel("Risk (σ)")
plt.ylabel("Expected Return E(R)")
plt.title("Efficient Frontier and Capital Allocation Line (CFA Level I)")
plt.legend()
plt.grid(True)

plt.show()
