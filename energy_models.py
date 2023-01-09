# Tutorial on DDEs: https://github.com/siavashBigdeli/DDE

# A neral network model for DDE needs to satisfy two cnoditions:
# 1. The output of the DDE is a scalar value.
# 2. The network needs to be twice-differentiable: i.e use activations that are everywhere differentiable (e.g. Sigmoid)

from tensorflow import keras
import tensorflow as tf

class DDE(keras.Model):
    def __init__(self, net_model):
        super(DDE, self).__init__()
        self.net_model = net_model

    def compile(self, optimizer):
        super(DDE, self).compile()
        self.optimizer = optimizer

    # Replace the call function of the model to include the gradient computation
    def call(self, noisy_input):
        # 1. Get the discriminator output for this interpolated image.
        with tf.GradientTape() as gp_tape:
            gp_tape.watch(noisy_input)
            log_prob = self.net_model(noisy_input, training=True)

        # 2. Calculate the gradients w.r.t to the input.
        grad_log_prob = gp_tape.gradient(log_prob, [noisy_input])[0]
                
        return log_prob, grad_log_prob

    # Calculate the noise estimation penalty based on MSE (Eq. 5 in paper)
    def noise_estimation_penalty(self, noisy_input, gt):
        """ Calculates the denoising penalty.
        """
        _, noise_estim = self.call(noisy_input) # compute the noise as the gradients of the network
        
        # 3. Calcuate the loss
        loss = tf.keras.losses.mean_squared_error(noise_estim, gt)
        
        return loss

    # Calculate model gradients for optimization and take an optimizatino step.
    # TF will implicitly include the loss gradients by going through the noise-estimation branch.
    def train_step(self, input):
        noisy_input = input[0]
        gt = input[1]

        # Get the batch size
        batch_size = tf.shape(noisy_input)[0]

        with tf.GradientTape() as tape:
            loss = self.noise_estimation_penalty(noisy_input, gt)

        # Get the gradients w.r.t the generator loss
        gradient = tape.gradient(loss, self.net_model.trainable_variables)
        # Update the weights of the generator using the generator optimizer
        self.optimizer.apply_gradients(
            zip(gradient, self.net_model.trainable_variables)
        )
        return {"loss": loss}
