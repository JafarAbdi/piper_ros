import numpy as np
from dataclasses import dataclass, field


@dataclass
class PIDController:
    """A Proportional-Integral-Derivative controller with anti-windup protection and filtered derivative."""

    # PID gains
    kp: float = 0.0
    ki: float = 0.0
    kd: float = 0.0

    # Controller parameters
    alpha: float = (
        0.75  # Derivative filter smoothing factor (0-1, lower = more filtering)
    )

    # Control effort bounds
    max_output: float = float("inf")

    # Internal state
    _set_point: float = 0.0
    _error_sum: float = 0.0
    _last_error: float = 0.0
    _filtered_derivative: float = 0.0  # Filtered derivative value

    def __post_init__(self):
        """Initialize derived attributes after dataclass initialization."""
        assert self.max_output > 0, "max_output must be positive"
        # Validate alpha is within valid range for filter
        if not 0 <= self.alpha <= 1:
            msg = "Alpha must be between 0 and 1"
            raise ValueError(msg)

    def __repr__(self) -> str:
        """Return a string representation of the PID controller."""
        return (
            f"PIDController(kp={self.kp}, ki={self.ki}, "
            f"kd={self.kd}, alpha={self.alpha}, "
            f"output_limits={self.output_limits})"
        )

    def reset(self) -> None:
        """Reset controller to initial state."""
        self._set_point = 0.0
        self._error_sum = 0.0
        self._last_error = 0.0
        self._filtered_derivative = 0.0

    def set_target(self, target: float) -> None:
        """Set the controller's target value (setpoint)."""
        self._set_point = float(target)

    def compute(self, measured_value: float) -> float:
        """Update the PID controller with a new measurement.

        Args:
            measured_value: The current measured process value
            debug: Publish debug info

        Returns:
            Control output value within defined limits
        """
        # Calculate error
        error = self._set_point - measured_value

        # Calculate derivative term
        delta_error = error - self._last_error

        # Apply low-pass filter to derivative term
        # alpha=1 means no filtering (use raw derivative)
        # alpha=0 means complete filtering (use previous filtered value)
        self._filtered_derivative = (
            self.alpha * delta_error + (1 - self.alpha) * self._filtered_derivative
        )

        # Update last error for next iteration
        self._last_error = error

        # Calculate PID components
        proportional_term = self.kp * error
        integral_term = self.ki * self._error_sum
        derivative_term = self.kd * self._filtered_derivative

        # Calculate total control output
        output = proportional_term + integral_term + derivative_term

        # Check for control signal saturation
        if abs(output) >= self.max_output:
            # Saturate the control signal
            output = np.sign(output) * self.max_output
        else:
            # Only integrate if control signal is not saturated (anti-windup)
            self._error_sum += error

        return output
