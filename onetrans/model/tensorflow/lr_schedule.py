import tensorflow as tf


class LinearWarmup(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, initial_learning_rate, peak_learning_rate, warmup_steps):
        super(LinearWarmup, self).__init__()
        self.initial_learning_rate = initial_learning_rate
        self.peak_learning_rate = peak_learning_rate
        self.warmup_steps = warmup_steps

    def __call__(self, step):
        scale = step / self.warmup_steps
        scale = tf.minimum(scale, 1.0)
        return (
            self.initial_learning_rate
            + (self.peak_learning_rate - self.initial_learning_rate) * scale
        )

    def get_config(self):
        return {
            "initial_learning_rate": self.initial_learning_rate,
            "peak_learning_rate": self.peak_learning_rate,
            "warmup_steps": self.warmup_steps,
        }
