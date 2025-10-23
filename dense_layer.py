
import numpy as np
from nnfs.datasets import spiral_data



class Layer_Dense:

	def __init__( self, n_inputs, n_neurons ):
		self.weights = np.random.randn( n_inputs, n_neurons )
		self.biases = np.zeros( ( 1, n_neurons ) )

	def forward( self, inputs ):
		self.inputs = inputs
		self.output = np.dot( inputs, self.weights ) + self.biases

	#dvalues is derivative we got from previous neurons
	def backward( self, dvalues ):
		self.dweights = np.dot( self.inputs.T, dvalues )
		self.dbiases = np.sum( dvalues, axis=0, keepdims=True )
		self.dinputs = np.dot( dvalues, self.weights.T )


class ReLU_Activation:

	def forward( self, inputs ):
		self.inputs = inputs
		self.output = np.maximum( 0, inputs )

	def backward( self, dvalues ):
		self.dinputs = dvalues.copy()
		self.dinputs[ self.inputs <= 0 ] = 0


class Softmax_Activation:

	def forward( self, inputs ):
		self.exp_value = np.exp( inputs - np.max( inputs, axis=1, keepdims=True ) )
		self.output = self.exp_value / np.sum( self.exp_value, axis=1, keepdims=True )

	def backward( self, dvalues ):

		self.dinputs = np.empty_like( dvalues )

		for index, ( single_output, single_dvalue ) in enumerate( zip( self.output, dvalues ) ):
			single_output = single_output.reshape( 1, -1 )
			jacobian_matrix = np.diagflat( single_output ) - np.dot( single_output, single_output.T )
			self.dinputs[ index ] = np.dot( jacobian_matrix, single_dvalue )


class Loss:

	def calculate( self, y_pred, y_actual ):
		sample_losses = self.forward( y_pred, y_actual )
		data_loss = np.mean( sample_losses )
		return data_loss


class Categorical_Cross_Entropy_Loss( Loss ):

	def forward( self, y_pred, y_actual ):
		samples = len( y_pred )
		y_pred_clipped = np.clip( y_pred, 1e-7, 1 - 1e-7 )
		correct_confidences = []
		if len( y_actual.shape ) == 1:
			correct_confidences = y_pred_clipped[ range( samples ), y_actual ]
		elif len( y_actual.shape ) == 2:
			correct_confidences = np.sum( y_pred_clipped * y_actual, axis=1 )
		negative_log = -np.log( correct_confidences )
		return negative_log

	def backward( self, dvalues, y_true ):
		samples = len( dvalues )
		labels = len( dvalues[ 0 ] )

		if len( y_true.shape ) == 1:
			y_true = np.eye( labels )[ y_true ]
		self.dinputs = -y_true / dvalues
		self.dinputs = self.dinputs / samples


class Softmax_Activation_Categorical_Cross_Entropy_Loss:

	def __init__( self ) -> None:
		self.activation = Softmax_Activation()
		self.loss = Categorical_Cross_Entropy_Loss()

	def forward( self, inputs, y_actual ):
		self.activation.forward( inputs )
		self.output = self.activation.output
		return self.loss.calculate( self.output, y_actual )

	def backward( self, dvalues, y_actual ):
		samples = len( dvalues )
		if len( y_actual.shape ) == 2:
			y_actual = np.argmax( y_actual, axis=1 )
		self.dinputs = dvalues.copy()
		self.dinputs[ range( samples ), y_actual ] -= 1
		self.dinputs = self.dinputs / samples


class SGD_Optimizer:

	def __init__( self, learning_rate=1., decay=0., momentum=0. ):
		self.learning_rate = learning_rate
		self.current_learning_rate = learning_rate
		self.decay = decay
		self.momentum = momentum
		self.iterations = 0

	def pre_update_params( self ):
		if self.decay:
			self.current_learning_rate = self.learning_rate * ( 1.0 / ( 1.0 + self.decay * self.iterations ) )

	def update_params( self, layer ):
		if self.momentum:
			if not hasattr( layer, "weight_momentums" ):
				layer.weight_momentums = np.zeros_like( layer.weights )
				layer.bias_momentums = np.zeros_like( layer.biases )
			weight_update = self.momentum * layer.weight_momentums - self.current_learning_rate * layer.dweights
			layer.weight_momentums = weight_update

			bias_update = self.momentum * layer.bias_momentums - self.current_learning_rate * layer.dbiases
			layer.bias_momentums = bias_update
		else:
			weight_update = -self.current_learning_rate * layer.dweights
			bias_update = -self.current_learning_rate * layer.dbiases

		layer.weights += weight_update
		layer.biases += bias_update

	def post_update_params( self ):
		self.iterations += 1


class AdaGrad_Optimizer:

	def __init__( self, learning_rate=1., decay=0., epsilon=1e-7 ):
		self.learning_rate = learning_rate
		self.current_learning_rate = learning_rate
		self.decay = decay
		self.epsilon = epsilon
		self.iterations = 0

	def pre_update_params( self ):
		if self.decay:
			self.current_learning_rate = self.learning_rate * ( 1.0 / ( 1.0 + self.decay * self.iterations ) )

	def update_params( self, layer ):

		if not hasattr( layer, "weight_cache" ):
			layer.weight_cache = np.zeros_like( layer.weights )
			layer.bias_cache = np.zeros_like( layer.biases )
		layer.weight_cache = layer.dweights ** 2

		layer.bias_cache = layer.dbiases ** 2

		layer.weights += -self.current_learning_rate * layer.dweights / ( np.sqrt( layer.weight_cache ) + self.epsilon )
		layer.biases += -self.current_learning_rate * layer.dbiases / ( np.sqrt( layer.bias_cache ) + self.epsilon )

	def post_update_params( self ):
		self.iterations += 1


X, y = spiral_data( samples=100, classes=3 )

dense1 = Layer_Dense( 2, 64 )
activation1 = ReLU_Activation()
dense2 = Layer_Dense( 64, 3 )
activation2 = Softmax_Activation_Categorical_Cross_Entropy_Loss()
optimizer = AdaGrad_Optimizer( decay=1e-4 )

for epoch in range( 10001 ):

	dense1.forward( X )
	activation1.forward( dense1.output )
	dense2.forward( activation1.output )
	loss = activation2.forward( dense2.output, y )

	prediction = np.argmax( activation2.output, axis=1 )
	if len( y.shape ) == 2:
		y = np.argmax( y, axis=1 )
	accuracy = np.mean( prediction == y )
	if not epoch % 100:
		print(
		    f"Epoch : {epoch}, Acc : {accuracy:.3f}, Loss : {loss:.3f}, Learning Rate : {optimizer.current_learning_rate}"
		    )

	activation2.backward( activation2.output, y )
	dense2.backward( activation2.dinputs )
	activation1.backward( dense2.dinputs )
	dense1.backward( activation1.dinputs )

	optimizer.pre_update_params()
	optimizer.update_params( dense1 )
	optimizer.update_params( dense2 )
	optimizer.post_update_params()
