import numpy as np

class CustomLSTMCell:
    def __init__(self, input_size, hidden_size, weights, weights_recurrent, biases):
        self.input_size = input_size
        self.hidden_size = hidden_size

        # Input gate weights and biases
        input_start, input_end = 0, hidden_size
        self.W_ix = weights[:, input_start:input_end].T
        self.W_ih = weights_recurrent[:, input_start:input_end].T
        self.b_i = biases[input_start:input_end]

        # Forget gate weights and biases
        forget_start, forget_end = hidden_size, hidden_size * 2
        self.W_fx = weights[:, forget_start:forget_end].T
        self.W_fh = weights_recurrent[:, forget_start:forget_end].T
        self.b_f = biases[forget_start:forget_end]

        # Cell state weights and biases
        cell_start, cell_end = hidden_size * 2, hidden_size * 3
        self.W_cx = weights[:, cell_start:cell_end].T
        self.W_ch = weights_recurrent[:, cell_start:cell_end].T
        self.b_c = biases[cell_start:cell_end]

        # Output gate weights and biases
        output_start, output_end = hidden_size * 3, hidden_size * 4
        self.W_ox = weights[:, output_start:output_end].T
        self.W_oh = weights_recurrent[:, output_start:output_end].T
        self.b_o = biases[output_start:output_end]

        # Initialize hidden state and cell state
        self.h_t = np.zeros((hidden_size, ))
        self.c_t = np.zeros((hidden_size, ))
   
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def tanh(self, x):
        return np.tanh(x)

    def forward(self, x_t):

        # Input gate
        i_t = self.sigmoid(np.dot(self.W_ix, x_t) + np.dot(self.W_ih, self.h_t) + self.b_i)

        # Forget gate
        f_t = self.sigmoid(np.dot(self.W_fx, x_t) + np.dot(self.W_fh, self.h_t) + self.b_f)

        # Cell state update
        tilde_c_t = self.tanh(np.dot(self.W_cx, x_t) + np.dot(self.W_ch, self.h_t) + self.b_c)
        self.c_t = f_t * self.c_t + i_t * tilde_c_t

        # Output gate
        o_t = self.sigmoid(np.dot(self.W_ox, x_t) + np.dot(self.W_oh, self.h_t) + self.b_o)
  
        # Hidden state update
        self.h_t = o_t * self.tanh(self.c_t)

        return self.h_t, self.c_t

class DenseLayer:
    def __init__(self, input_size, output_size, weights, biases):
        self.input_size = input_size
        self.output_size = output_size

        # Initialize weights and biases
        self.W = weights.T
        self.b = biases

    def forward(self, x):
        z = np.dot(self.W, x) + self.b
        exp_z = np.exp(z - np.max(z))
        return exp_z / np.sum(exp_z, axis=0, keepdims=True)

class CustomLSTM:
    def __init__(self, input_size, hidden_size, weights, weights_recurrent, biases):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.cell = CustomLSTMCell(input_size, hidden_size, weights, weights_recurrent, biases)
    
    def forward(self, x):
        seq_len, _ = x.shape
        for t in range(seq_len):
            hidden_state, _ = self.cell.forward(x[t])
        return hidden_state

class CustomLSTMModel:
    def __init__(self, input_shape, units, num_classes, weights):
        self.input_shape = input_shape
        self.units = units
        self.num_classes = num_classes

        self.lstm_layer = CustomLSTM(input_shape[1], units, weights[0], weights[1], weights[2])
        self.dense_layer = DenseLayer(units, num_classes, weights[3], weights[4])

    def forward(self, x):
        lstm_output = self.lstm_layer.forward(x)
        # Use the output of the last timestep for classification
        return self.dense_layer.forward(lstm_output)

