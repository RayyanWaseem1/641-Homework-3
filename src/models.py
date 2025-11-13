import torch
import torch.nn as nn

class SentimentRNN(nn.Module):
    """Basic RNN for sentiment classification"""

    def __init__(self, vocab_size, embedding_dim = 100, hidden_dim = 64, num_layers = 2, dropout = 0.4, activation = "tanh"):
        """
        initialize the RNN
        Args:
            vocab_size (int): size of vocabulary
            embedding_dim (int): dimension of word embeddings
            hidden_dim (int): dimension of RNN hidden states
            num_layers (int): number of RNN layers
            dropout (float): dropout rate
            activation (str): activation function ("tanh" or "relu")
        """
        super(SentimentRNN, self).__init__()

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.activation_name = activation 

        #Embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx = 0)

        #RNN layer
        if activation == "tanh":
            nonlinearity = "tanh"
        elif activation == "relu":
            nonlinearity = "relu"
        else:
            nonlinearity = "tanh"

        self.rnn = nn.RNN(
            embedding_dim,
            hidden_dim,
            num_layers = num_layers,
            batch_first = True, 
            dropout = dropout if num_layers > 1 else 0,
            nonlinearity = nonlinearity
        )

        #Dropout layer
        self.dropout = nn.Dropout(dropout)

        #Fully connected output layer
        self.fc = nn.Linear(hidden_dim, 1)

        #Output activation
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """
        Forward pass of RNN
        
        Args:
            x (torch.Tensor): input tensor of shape (batch_size, seq_length)
        Returns:
            torch.Tensor: output probavilities of the shape (batch_size, 1)
        """

        #Embedding: (batch size, seq_length, embedding_dim)
        embedded = self.embedding(x)

        #RNN: output shape (batch_size, seq_length, hidden_dim)
        #hidden layer (num_layers, batch_size, hidden_dim)
        rnn_out, hidden = self.rnn(embedded)

        #Get the last time step output
        last_out = rnn_out[:, -1, :]

        #applying dropout
        last_out = self.dropout(last_out)

        #Fully connected layer
        out = self.fc(last_out)

        #sigmoid activation
        out = self.sigmoid(out)

        return out.squeeze(1)
    

class SentimentLSTM(nn.Module):
    """ 
    LSTM for Sentiment Classification
    """

    def __init__(self, vocab_size, embedding_dim = 100, hidden_dim = 64, num_layers = 2, dropout = 0.4, activation = "tanh"):

        """
        Args:
            vocab_size (int): size of vocabulary
            embedding_dim (int): dimension of embeddings
            hidden_dim (int): dimension of hidden layer
            num_layers (int): number of LSTM layers
            dropout (float): dropout probability
            activation (str): Activation function (for fc layer)
            
        """

        super(SentimentLSTM, self).__init__()

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers 
        self.activation_name = activation 

        #embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx = 0)

        #LSTM layer
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            num_layers = num_layers,
            batch_first = True,
            dropout = dropout if num_layers > 1 else 0
        )
        
        #Dropout layer
        self.dropout = nn.Dropout(dropout)

        #activation function for FC layer
        if activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "sigmoid":
            self.activation = nn.Sigmoid()
        else:
            self.activation = nn.Tanh()

        #Fully connected output layer
        self.fc = nn.Linear(hidden_dim, 1)

        #Output activation
        self.output_activation = nn.Sigmoid() 

    def forward(self, x):
        """
        Forward pass

        Args:
            x (torch.Tensor): input tensor of shape (batch_size, seq_length)

        Returns:
            torch.Tensor: output probabilities of shape (batch_size, 1)
        """

        #embeddings: (batch_size, seq_length, embedding_dim)
        embedded = self.embedding(x)

        #LSTM: output shape (batch_size, seq_length, hidden_dim)
        lstm_out, (hidden, cell) = self.lstm(embedded)

        #take the last time step output
        last_out = lstm_out[:, -1, :]

        #applying dropout
        last_out = self.dropout(last_out)

        #applying activation
        last_out = self.activation(last_out)

        #fully connected layer 
        out = self.fc(last_out)

        #sigmoid activation
        out = self.output_activation(out)

        return out.squeeze(1)
    
class SentimentBiLSTM(nn.Module):

    """
    Bidirectional LSTM for sentiment classification
    """

    def __init__(self, vocab_size, embedding_dim = 100, hidden_dim = 64, num_layers = 2, dropout = 0.4, activation = "tanh"):
        """
        Initializing BiLSTM
        
        Args:
            vocab_size (int): size of vocabulary
            embedding_dim (int): dimension of embeddings
            hidden_dim (int): dimension of hidden layer
            num_layers (int): number of LSTM layers
            dropout (float): dropout probability
            activation (str): Activation function (for fc layer)
        """

        super(SentimentBiLSTM, self).__init__()

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.activation_name = activation

        #embedding layer
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx = 0)

        #bidirectional LSTM layer
        self.lstm = nn.LSTM(
            embedding_dim,
            hidden_dim,
            num_layers = num_layers,
            batch_first = True,
            dropout = dropout if num_layers > 1 else 0,
            bidirectional = True
        )

        #dropout layer
        self.dropout = nn.Dropout(dropout)

        #activation function for FC layer
        if activation == "relu":
            self.activation = nn.ReLU()
        elif activation == "sigmoid":
            self.activation = nn.Sigmoid()
        else:
            self.activation = nn.Tanh()

        #fully connected output layer (input is 2 * hidden_dim due to bidirectionality)
        self.fc = nn.Linear(hidden_dim * 2, 1)

        #output activation
        self.output_activation = nn.Sigmoid()

    def forward(self, x):
        """
        forward pass.

        Args:
            x (torch.Tensor): input tensor of shape (batch_size, seq_length)
        Returns:
            torch.Tensor: output probabilities of shape (batch_size, 1)
        """

        embedded = self.embedding(x)

        #LSTM: output shape (batch_size, seq_length, hidden_dim * 2)
        lstm_out, (hidden, cell) = self.lstm(embedded)

        #take the last time step output
        last_out = lstm_out[:, -1, :]

        #apply dropout 
        last_out = self.dropout(last_out)

        #apply activation
        last_out = self.activation(last_out)

        #fully connected layer
        out = self.fc(last_out)

        #sigmoid activation
        out = self.output_activation(out)

        return out.squeeze(1)
    
def get_model(model_type, vocab_size, embedding_dim = 100, hidden_dim = 64, num_layers = 2, dropout = 0.4, activation = "tanh"):
    """
    Utility function to get the model based on type

    Args:
        model_type (str): type of model ("rnn", "lstm", "bilstm")
        vocab_size (int): size of vocabulary
        embedding_dim (int): dimension of embeddings
        hidden_dim (int): dimension of hidden layer
        num_layers (int): number of layers
        dropout (float): dropout probability
        activation (str): activation function

    Returns:
        nn.Module: instantiated model
    """

    if model_type == "rnn":
        return SentimentRNN(
            vocab_size,
            embedding_dim,
            hidden_dim,
            num_layers,
            dropout,
            activation)
    elif model_type == "lstm":
        return SentimentLSTM(
            vocab_size, 
            embedding_dim, 
            hidden_dim, 
            num_layers, 
            dropout, 
            activation)
    elif model_type == "bilstm":
        return SentimentBiLSTM(
            vocab_size, 
            embedding_dim, 
            hidden_dim, 
            num_layers, 
            dropout, 
            activation)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    return model 


if __name__ == "__main__":
    vocab_size = 10000
    batch_size = 32
    seq_length = 50

    #Dummy input
    dummy_input = torch.randint(0, vocab_size, (batch_size, seq_length))

    #testing RNN
    print("Testing RNN ")
    rnn_model = get_model('rnn', vocab_size)
    rnn_output = rnn_model(dummy_input)
    print(f"RNN output shape: {rnn_output.shape}")

    #testing LSTM
    print("Testing LSTM ")
    lstm_model = get_model('lstm', vocab_size)
    lstm_output = lstm_model(dummy_input)
    print(f"LSTM output shape: {lstm_output.shape}")

    #testing BiLSTM
    print("Testing BiLSTM ")
    bilstm_model = get_model('bilstm', vocab_size)
    bilstm_output = bilstm_model(dummy_input)
    print(f"BiLSTM output shape: {bilstm_output.shape}")


    print("All models tested successfully")
    
        
