class KalmanSmoother:
    def __init__(self, sigma2_w=1e-3, sigma2_v=1e-1):
        self.Q = sigma2_w
        self.R = sigma2_v
        self.reset()

    def reset(self):
        self.s_hat = 0.5
        self.P = 1.0

    def update(self, p_t):
        s_pred = self.s_hat
        P_pred = self.P + self.Q
        K = P_pred / (P_pred + self.R)
        self.s_hat = s_pred + K * (p_t - s_pred)
        self.P = (1 - K) * P_pred
        return self.s_hat

    def smooth_sequence(self, probs):
        self.reset()
        return [self.update(float(p)) for p in probs]
