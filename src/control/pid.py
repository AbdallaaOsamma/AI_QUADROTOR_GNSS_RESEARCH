# src/control/pid.py
from dataclasses import dataclass

@dataclass
class PIDGains:
    kp: float
    ki: float
    kd: float
    limit: float

class PID:
    def __init__(self, gains: PIDGains):
        self.g = gains
        self.i = 0.0
        self.prev_err = 0.0
        self._first = True

    def reset(self):
        self.i = 0.0
        self.prev_err = 0.0
        self._first = True

    def update(self, error: float, dt: float) -> float:
        if dt <= 0:
            dt = 1e-3
        # integral with simple anti-windup clamp
        self.i += error * dt
        self.i = max(min(self.i, self.g.limit), -self.g.limit)
        # derivative (protect first step)
        d = 0.0 if self._first else (error - self.prev_err) / dt
        self._first = False
        self.prev_err = error
        # PID sum
        u = self.g.kp * error + self.g.ki * self.i + self.g.kd * d
        # output clamp
        return max(min(u, self.g.limit), -self.g.limit)
