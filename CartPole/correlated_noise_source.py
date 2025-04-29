import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm


class CorrelatedNoiseGenerator:
    def __init__(self, rng, bias=0.0, sigma=1.0, a=0.9, dt=1.0, initial_state=0.0):
        """
        Initialize the correlated noise generator with time step adjustment.

        Parameters:
            a (float): Base correlation coefficient for a time step of 1.
                       The effective correlation for a given dt is computed as a_eff = a**(dt),
                       ensuring the physical correlation remains invariant when dt changes.
            sigma (float): Standard deviation of the underlying Gaussian noise.
            dt (float): The size of a single time step.
            initial_state (float): The starting value for the noise process.
            rng (np.random.Generator): Random number generator for reproducibility.
        """
        self.rng = rng
        self.bias = bias
        self.sigma = sigma
        self.a = a
        self.dt = dt
        # Compute effective AR(1) coefficient given the time step.
        self.effective_a = self.a ** self.dt
        self.last_sample = initial_state

    def sample(self, size=1):
        """
        Generate the next correlated noise sample using an AR(1) process adjusted for dt.

        The update is given by:
            x[n+1] = a_eff * x[n] + sqrt(1 - a_eff^2) * e[n]
        where a_eff = a**(dt) and e[n] ~ N(0, sigma^2).

        Returns:
            float: The next noise sample (scalar when size == 1).

        Note:
            The AR(1) process is inherently sequential. Thus, vectorized sampling (size > 1)
            is not supported. If you need vectorized output, you would have to iterate the recurrence.
        """

        # Obtain a scalar from the random generator.
        white_noise = self.sigma * self.rng.standard_normal(size=size) + self.bias
        sample = self.effective_a * self.last_sample + np.sqrt(1 - self.effective_a ** 2) * white_noise
        # Update the process state with the new scalar value.
        self.last_sample = sample
        return sample

    def reset(self, s0=None, dt=None):
        """
        Reset the internal state of the correlated noise generator.

        Parameters:
            s0 (float): The new initial state for the noise process.
            dt (float): The new time step. If None, the current dt is retained.
        """
        if s0 is not None:
            self.last_sample = s0

        if dt is not None:
            # Update the time step and recompute the effective correlation.
            self.dt = dt
            self.effective_a = self.a ** self.dt



def compute_autocorrelation(signal):
    """
    Compute the normalized autocorrelation for each coordinate (independent time series)
    in a multidimensional signal.

    Assumptions:
        - The signal is a 2D array of shape (n_timesteps, n_dimensions).
        - Each column corresponds to an independent time series.
        - If a 1D signal is provided, it is assumed to be a single time series and converted accordingly.

    Parameters:
        signal (np.ndarray): Input signal with shape (n_timesteps,) or (n_timesteps, n_dimensions).

    Returns:
        np.ndarray: Autocorrelation functions for all coordinates (shape:
                    (n_timesteps, n_dimensions)), where each column is normalized
                    so that the zero-lag autocorrelation equals 1.
    """
    # If signal is 1D, convert it to a 2D array with one column.
    if signal.ndim == 1:
        signal = signal[:, None]

    # Determine the number of timesteps (n) and dimensions (d)
    n, d = signal.shape

    # Center the data for each coordinate by subtracting its mean.
    signal_centered = signal - np.mean(signal, axis=0)
    autocorrs = np.empty((n, d))

    # Process each independent time series (each dimension) individually.
    for i in range(d):
        autocorr_full = np.correlate(signal_centered[:, i], signal_centered[:, i], mode='full')
        autocorr = autocorr_full[n - 1:]  # Select lags >= 0.
        autocorr /= autocorr[0]           # Normalize so that lag 0 equals 1.
        autocorrs[:, i] = autocorr

    return autocorrs


# --- Demonstration of the time-step invariant noise generator ---
if __name__ == '__main__':
    # Define different time steps for demonstration.
    dt_values = [1.0, 0.5, 0.1]  # dt = 1, 0.5, and 0.1

    # Create a base RNG for reproducibility.
    base_rng = np.random.default_rng(42)
    # Generate independent seeds for each generator to ensure independent noise streams.
    seeds = base_rng.integers(low=0, high=10000, size=len(dt_values))
    generators = [
        CorrelatedNoiseGenerator(
            rng=np.random.default_rng(seed),  # Pass an RNG seeded independently.
            bias=0.0, sigma=1.0, a=0.9, dt=dt
        )
        for seed, dt in zip(seeds, dt_values)
    ]

    n_samples = 1000  # Number of samples per generator
    # Adjust the number of samples per generator based on the time step.
    noise_data = [[gen.sample() for _ in range(int(n_samples / gen.dt))] for gen in generators]

    # Set up a 3x3 grid of subplots: one row per noise source.
    fig, axes = plt.subplots(len(dt_values), 3, figsize=(18, 12))

    for i, (dt, samples) in enumerate(zip(dt_values, noise_data)):
        time_line = np.arange(len(samples)) * dt
        # Convert list of scalars to 1D numpy array.
        samples = np.array(samples)

        # --- Time Series Plot ---
        axes[i, 0].plot(time_line, samples, label=f'dt = {dt}')
        axes[i, 0].set_title(f'Time Series (dt = {dt})')
        axes[i, 0].set_xlabel('Time (s)')
        axes[i, 0].set_ylabel('Amplitude')
        axes[i, 0].legend()

        # --- Histogram with Gaussian PDF ---
        counts, bins, _ = axes[i, 1].hist(samples, bins=30, density=True, alpha=0.6, color='g', label='Histogram')
        x = np.linspace(bins[0], bins[-1], 100)
        axes[i, 1].plot(x, norm.pdf(x, 0, 1.0), 'r', label='Gaussian PDF')
        axes[i, 1].set_title(f'Histogram (dt = {dt})')
        axes[i, 1].set_xlabel('Amplitude')
        axes[i, 1].set_ylabel('Density')
        axes[i, 1].legend()

        # --- Autocorrelation Plot ---
        # Compute autocorrelation; if samples is 1D it becomes a 2D array with one column.
        autocorrs = compute_autocorrelation(samples)
        # Extract the full autocorrelation for the single time series.
        autocorr = autocorrs[:, 0]
        # Create lag indices for plotting.
        lags = np.arange(len(autocorr))
        axes[i, 2].plot(lags, autocorr, marker='o', linestyle='-', label='Autocorrelation')
        axes[i, 2].set_title(f'Autocorrelation (dt = {dt})')
        axes[i, 2].set_xlabel('Lag')
        axes[i, 2].set_ylabel('Autocorrelation')
        axes[i, 2].legend()

    plt.tight_layout()
    plt.show()
