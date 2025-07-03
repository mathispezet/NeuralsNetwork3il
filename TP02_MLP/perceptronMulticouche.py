import numpy as np
import matplotlib.pyplot as plt

class ActivationFunction:
    """
    Classe simple pour gérer les fonctions d'activation et leurs dérivées.
    """
    def __init__(self, name='sigmoid'):
        self.name = name

    def __call__(self, z):
        """ Calcule la valeur de la fonction d'activation. """
        if self.name == 'sigmoid':
            # np.clip évite les erreurs avec de très grandes/petites valeurs
            return 1 / (1 + np.exp(-np.clip(z, -500, 500)))
        return z # Pour une activation linéaire par défaut

    def derivative(self, z):
        """ Calcule la dérivée de la fonction d'activation. """
        if self.name == 'sigmoid':
            s = self.__call__(z)
            return s * (1 - s)
        return np.ones_like(z) # Pour une activation linéaire par défaut

class CoucheNeurones:
    """ Représente une seule couche de neurones. """
    def __init__(self, n_input, n_neurons, activation='sigmoid', learning_rate=0.01):
        self.learning_rate = learning_rate
        # Initialisation des poids (Xavier/Glorot) pour une meilleure convergence
        limit = np.sqrt(6 / (n_input + n_neurons))
        self.weights = np.random.uniform(-limit, limit, (n_neurons, n_input))
        self.bias = np.zeros((n_neurons, 1))
        # Garder en mémoire les valeurs pour la rétropropagation
        self.last_input = None
        self.last_z = None
        self.activation_func = ActivationFunction(activation)

    def forward(self, X):
        self.last_input = X
        self.last_z = np.dot(self.weights, X) + self.bias
        return self.activation_func(self.last_z)

    def backward(self, gradient_from_next_layer):
        m = self.last_input.shape[1]
        delta_couche = gradient_from_next_layer * self.activation_func.derivative(self.last_z)
        grad_weights = (1/m) * np.dot(delta_couche, self.last_input.T)
        grad_bias = (1/m) * np.sum(delta_couche, axis=1, keepdims=True)
        grad_input = np.dot(self.weights.T, delta_couche)
        self.weights -= self.learning_rate * grad_weights
        self.bias -= self.learning_rate * grad_bias
        return grad_input


class PerceptronMultiCouches:
    def __init__(self, architecture, learning_rate=0.01, activation='sigmoid'):
        self.architecture = architecture
        self.couches = []
        self.history = {'loss': [], 'val_loss': [], 'accuracy': [], 'val_accuracy': []}
        for i in range(len(architecture) - 1):
            act_func = 'sigmoid' if i == len(architecture) - 2 else activation
            couche = CoucheNeurones(architecture[i], architecture[i+1], act_func, learning_rate)
            self.couches.append(couche)

    def forward(self, X):
        current_input = X.T
        for couche in self.couches:
            current_input = couche.forward(current_input)
        return current_input.T

    def backward(self, X, y_true, y_pred):
        gradient = (y_pred - y_true).T
        for couche in reversed(self.couches):
            gradient = couche.backward(gradient)

    def compute_loss(self, y_true, y_pred):
        return np.mean(np.square(y_true - y_pred))

    def compute_accuracy(self, y_true, y_pred):
        predictions = (y_pred > 0.5).astype(int)
        return np.mean(predictions.flatten() == y_true.flatten())
    
    def train_epoch(self, X, y):
        y_pred = self.forward(X)
        loss = self.compute_loss(y, y_pred)
        self.backward(X, y, y_pred)
        return loss, y_pred

    def fit(self, X, y, X_val=None, y_val=None, epochs=100, verbose=True):
        for epoch in range(epochs):
            loss, y_pred = self.train_epoch(X, y)
            accuracy = self.compute_accuracy(y, y_pred)
            self.history['loss'].append(loss)
            self.history['accuracy'].append(accuracy)
            if X_val is not None and y_val is not None:
                y_val_pred = self.predict(X_val)
                self.history['val_loss'].append(self.compute_loss(y_val, y_val_pred))
                self.history['val_accuracy'].append(self.compute_accuracy(y_val, y_val_pred))
            if verbose and (epoch % 1000 == 0 or epoch == epochs - 1):
                log_message = f"Époque {epoch:4d} - Loss: {loss:.4f} - Acc: {accuracy:.4f}"
                if 'val_loss' in self.history and self.history['val_loss']:
                    log_message += f" - Val Loss: {self.history['val_loss'][-1]:.4f} - Val Acc: {self.history['val_accuracy'][-1]:.4f}"
                print(log_message)

    def predict(self, X):
        return self.forward(X)

def plot_decision_boundary_mlp(mlp, X, y, title):
    """
    Trace la surface de décision d'un MLP (version simple).
    """
    plt.figure()

    # Créer une grille de points
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02), np.arange(y_min, y_max, 0.02))

    # Prédire sur la grille
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    Z = mlp.predict(grid_points)
    Z = (Z > 0.5).astype(int)
    Z = Z.reshape(xx.shape)

    # Tracer la surface (le "fond") et les points
    plt.contourf(xx, yy, Z, alpha=0.4)
    plt.scatter(X[:, 0], X[:, 1], c=y.flatten(), edgecolors='k')
    
    plt.title(title)

def plot_loss(history, title):
    """ Trace la courbe de perte (version simple). """
    plt.figure()
    plt.plot(history['loss'])
    if history['val_loss']:
        plt.plot(history['val_loss'])
    plt.title(title)
    plt.xlabel("Époques")
    plt.ylabel("Perte")

def plot_accuracy(history, title):
    """ Trace la courbe de précision (version simple). """
    plt.figure()
    plt.plot(history['accuracy'])
    if history['val_accuracy']:
        plt.plot(history['val_accuracy'])
    plt.title(title)
    plt.xlabel("Époques")
    plt.ylabel("Précision")

def test_xor_avec_plots_simplifies():
    """ Teste le réseau sur XOR et génère les visualisations simples. """
    X_xor = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y_xor = np.array([[0], [1], [1], [0]])

    print("Test sur le problème XOR avec visualisations simplifiées")
    
    arch = [2, 4, 1]
    print(f"Architecture {arch} ")
    
    np.random.seed(42)
    
    # Créer et entraîner le réseau
    mlp = PerceptronMultiCouches(arch, learning_rate=0.1, activation='sigmoid')
    mlp.fit(X_xor, y_xor, X_val=X_xor, y_val=y_xor, epochs=10000, verbose=True)

    print("Génération des graphiques...")
    
    # Générer les graphiques un par un
    plot_loss(mlp.history, title=f"Évolution de la Perte (Arch {arch})")
    plot_accuracy(mlp.history, title=f"Évolution de la Précision (Arch {arch})")
    plot_decision_boundary_mlp(mlp, X_xor, y_xor, title=f"Surface de Décision (Arch {arch})")

    # Afficher tous les graphiques
    plt.show()

# Lancer le test
if __name__ == "__main__":
    test_xor_avec_plots_simplifies()