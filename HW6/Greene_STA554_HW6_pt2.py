import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
from numpy.random import default_rng

# Create a class that encapsulates the simulation of the sampling dist and slope
class SLR_slope_simulator:
  # Initialize the class with arguments self, beta_0, beta_1, sigma, x, n, and seed
  def __init__(self, beta_0, beta_1, x, sigma, seed):
    self.beta_0 = beta_0
    self.beta_1 = beta_1
    self.sigma = sigma
    self.x = x
    self.n = len(x)
    self.rng = np.random.default_rng(seed)
    self.slopes = []

  # Create generate_data method
  def generate_data(self):
    y = self.beta_0 + self.beta_1 * self.x + self.rng.normal(0, self.sigma, self.n)
    return self.x, y

  # Create fit_slope method
  def fit_slope(self, x, y):
    reg = linear_model.LinearRegression()
    fit = reg.fit(x.reshape(-1, 1), y)
    return fit.coef_[0]

  # Create run_sim method
  def run_sim(self, n_reps):

      slopes = []

      for i in range(n_reps):
          x, y = self.generate_data()
          slope = self.fit_slope(x, y)
          slopes.append(slope)

      self.slopes = np.array(slopes)

  # Create plot_dist method
  def plot_dist(self):

    if len(self.slopes) == 0:
        print("Error! run_simulations() must be called first")
        return

    plt.hist(self.slopes, bins=20)
    plt.xlabel('Slope')
    plt.ylabel('Frequency')
    plt.title('Distribution of Slopes')
    plt.show()

  # Create find_prob method
  def find_prob(self, value, sided):

    if len(self.slopes) == 0:
        print("Error! run_simulations() must be called first")
        return

    if sided == "above":
        prob = np.mean(self.slopes > value)

    elif sided == "below":
        prob = np.mean(self.slopes < value)

    elif sided == "two-sided":
        prob = np.mean(np.abs(self.slopes) > abs(value))

    else:
        print("Invalid sided argument")
        return

    return prob

#  Below you definition of your class and its methods, add a section of code that

# 1. Creates an instance of the object with  beta_0  = 12,  beta_1  = 2,  x  =  np.array(list(np.linspace(start
# = 0, stop = 10, num = 11))*3) ,  sigma  = 1, and  seed  = 10
# 2. Call your  run_simulation()  method (this should return the error message)
# 3. Run 10000 simulations
# 4. Plot the sampling distribution
# 5. Approximate the two-sided probability of being larger than 2.1
# 6. Print out the value of the simulated slopes using the attribute

# Create instance of the class object
x = np.array(list(np.linspace(start = 0, stop = 10, num = 11))*3)

sim = SLR_slope_simulator(12, 2, x, 1, 10)

# This returns an error
sim.plot_dist()

sim.run_sim(10000)

sim.plot_dist()

prob = sim.find_prob(2.1, "two-sided")
print(prob)

print(sim.slopes)