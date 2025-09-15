from datasets import load_dataset
import numpy as np

def load_mnist_dataset():
    train_data = load_dataset("ylecun/mnist", split="train")
    test_data = load_dataset("ylecun/mnist", split="test")
    
    X_train = np.array([np.array(img['image']).flatten() for img in train_data])
    y_train = np.array([img['label'] for img in train_data])
    X_test = np.array([np.array(img['image']).flatten() for img in test_data])
    y_test = np.array([img['label'] for img in test_data])
    
    return X_train, y_train, X_test, y_test

def preprocess_data(images, labels, flatten=True, normalize=True):
    processed_images = images.astype(np.float32)
    
    if normalize:
        processed_images = processed_images / 255.0
    
    if flatten:
        processed_images = processed_images.reshape(images.shape[0], -1)
    
    return processed_images, labels

class NeuralNetwork:
    def __init__(self, input_size, hidden_sizes, output_size, learning_rate=0.01, dropout_rate=0.2):
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes
        self.output_size = output_size
        self.learning_rate = learning_rate
        self.dropout_rate = dropout_rate
        
        self.W1 = np.random.randn(self.input_size, self.hidden_sizes) * 0.01
        self.b1 = np.zeros((1, self.hidden_sizes))
        self.W2 = np.random.randn(self.hidden_sizes, self.output_size) * 0.01
        self.b2 = np.zeros((1, self.output_size))
        
        self.v_W1 = np.zeros_like(self.W1)
        self.v_b1 = np.zeros_like(self.b1)
        self.v_W2 = np.zeros_like(self.W2)
        self.v_b2 = np.zeros_like(self.b2)
        self.beta = 0.9  # Momentum hyperparameter
    
    def softmax(self, x):
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)
    
    def relu(self, x):
        return np.maximum(0, x)
    
    def relu_derivative(self, x):
        return (x > 0).astype(float)
    
    def dropout(self, x, training=True):
        if training:
            mask = np.random.binomial(1, 1-self.dropout_rate, x.shape) / (1-self.dropout_rate)
            return x * mask
        return x
    
    def forward(self, X, training=True):
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = self.relu(self.z1)
        self.a1_dropped = self.dropout(self.a1, training)
        self.z2 = np.dot(self.a1_dropped, self.W2) + self.b2
        self.a2 = self.softmax(self.z2)
        return self.a2
    
    def backward(self, X, y, output):
        m = X.shape[0]
        
        dz2 = output - y
        dW2 = (1/m) * np.dot(self.a1_dropped.T, dz2)
        db2 = (1/m) * np.sum(dz2, axis=0, keepdims=True)
        
        da1_dropped = np.dot(dz2, self.W2.T)
        
        da1 = da1_dropped
        dz1 = da1 * self.relu_derivative(self.z1)
        dW1 = (1/m) * np.dot(X.T, dz1)
        db1 = (1/m) * np.sum(dz1, axis=0, keepdims=True)
        
        self.v_W2 = self.beta * self.v_W2 + (1 - self.beta) * dW2
        self.v_b2 = self.beta * self.v_b2 + (1 - self.beta) * db2
        self.v_W1 = self.beta * self.v_W1 + (1 - self.beta) * dW1
        self.v_b1 = self.beta * self.v_b1 + (1 - self.beta) * db1
        
        self.W2 -= self.learning_rate * self.v_W2
        self.b2 -= self.learning_rate * self.v_b2
        self.W1 -= self.learning_rate * self.v_W1
        self.b1 -= self.learning_rate * self.v_b1
    
    
    def compute_loss(self, y_true, y_pred):
        m = y_true.shape[0]
        loss = -np.sum(y_true * np.log(y_pred + 1e-8)) / m
        return loss
    
    def one_hot_encode(self, labels, num_classes):
        one_hot = np.zeros((labels.shape[0], num_classes))
        one_hot[np.arange(labels.shape[0]), labels] = 1
        return one_hot
    
    def train(self, X, y, epochs, batch_size=32):
        num_samples = X.shape[0]
        num_classes = self.output_size
        y_one_hot = self.one_hot_encode(y, num_classes)
        
        lossed = []
        accuracies = []
        
        for epoch in range(epochs):
            epoch_loss = 0
            correct_predicitons = 0
            
            for i in range(0, num_samples, batch_size):
                batch_X = X[i:i+batch_size]
                batch_y = y_one_hot[i:i+batch_size]
                
                output = self.forward(batch_X, training=True)
                batch_loss = self.compute_loss(batch_y, output)
                epoch_loss += batch_loss
                
                pred = np.argmax(output, axis=1)
                true_lables = np.argmax(batch_y, axis=1)
                correct_predicitons += np.sum(pred == true_lables)
                
                self.backward(batch_X, batch_y, output)
            
            avg_loss = epoch_loss / (num_samples // batch_size)
            accuracy = correct_predicitons / num_samples
            
            lossed.append(avg_loss)
            accuracies.append(accuracy)
            
            if epoch % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}")
        
        return lossed, accuracies
    
    def predict(self, X):
        output = self.forward(X, training=False)
        return np.argmax(output, axis=1)
    
    def evaluate(self, X, y):
        pred = self.predict(X)
        accuracy = np.mean(pred == y)
        return accuracy

if __name__ == "__main__":
    try:
        from datasets import load_dataset
        print("Using Hugging Face datasets library...")
        X_train, y_train, X_test, y_test = load_mnist_dataset()
        
        if X_train is not None:
            X_train, y_train = preprocess_data(X_train, y_train)
            X_test, y_test = preprocess_data(X_test, y_test)
            
            nn = NeuralNetwork(784, 128, 10, learning_rate=0.01, dropout_rate=0.2)
            losses, accuracies = nn.train(X_train, y_train, epochs=50, batch_size=64)
            
            test_accuracy = nn.evaluate(X_test, y_test)
            print(f"Final test accuracy: {test_accuracy:.4f}")
    except ImportError:
        print("Hugging Face datasets not available, using manual download...")