import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

class PerceptronSimple:
    def __init__(self, learning_rate=0.1):
        self.learning_rate = learning_rate
        self.weights = None
        self.bias = None

    #entraine le modèle puis retourne l'historique des erreurs
    def fit(self, X, y, max_epochs=100):
        """
        Entraîne le perceptron et retourne l'historique des erreurs.
        """
        # Initialisation des poids et du biais
        self.weights = np.random.randn(X.shape[1])
        self.bias = 0.0
        
        error_history = []

        for e in tqdm(range(max_epochs), desc=f"Training (lr={self.learning_rate})"):
            
            # on mesure l'erreur avec les poids de l'état actuel
            current_errors = np.sum(self.predict(X) != y)
            error_history.append(current_errors)

            # on entraîne pour l'époque, ce qui modifiera les poids pour la prochaine itération
            for i in range(X.shape[0]):
                x = X[i]
                y_true = y[i]
                
                prediction_bool = (np.dot(self.weights, x) + self.bias) >= 0
                y_pred = int(prediction_bool)
    
                error = y_true - y_pred
                self.weights += self.learning_rate * error * x
                self.bias += self.learning_rate * error   
        
        return error_history
                
    def predict(self, X):
        """Prédit les sorties pour les entrées X"""
        y_pred = np.zeros(X.shape[0])
        for i in range(X.shape[0]):
            x = X[i]
            prediction_bool = (np.dot(self.weights, x) + self.bias) >= 0
            y_pred[i] = int(prediction_bool)
        return y_pred

    def score(self, X, y):
        """Calcule l'accuracy"""
        predictions = self.predict(X)
        return np.mean(predictions == y)    
    
def plot_decision_boundary(perceptron, X, y, title):
    plt.figure()
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap='winter', edgecolors='k')
    w = perceptron.weights
    b = perceptron.bias
    x_values = np.array([X[:, 0].min() - 1, X[:, 0].max() + 1])
    if w[1] != 0:
        y_values = -(w[0] * x_values + b) / w[1]
        plt.plot(x_values, y_values, 'r-')
    else:
        x_line = -b / w[0]
        plt.axvline(x=x_line, color='r')
    plt.xlim(x_values)
    plt.title(title)    
    plt.show()

def generer_donnees_separables(n_points=100, noise=0.2):

    n_per_class = n_points // 2
    X1 = np.random.randn(n_per_class, 2) * noise + np.array([2, 2])
    y1 = np.ones(n_per_class)
    X0 = np.random.randn(n_per_class, 2) * noise + np.array([-2, -2])
    y0 = np.zeros(n_per_class)
    X = np.vstack((X1, X0))
    y = np.concatenate((y1, y0))
    return X, y

def analyser_convergence(X, y, learning_rates, max_epochs=20):
    """
    Entraîne un perceptron pour différents taux d'apprentissage et trace les courbes de convergence.
    """
    plt.figure(figsize=(10, 7))
    
    for lr in learning_rates:
        # Crée un nouveau perceptron pour chaque taux d'apprentissage
        perceptron = PerceptronSimple(learning_rate=lr)
        
        # Entraîne le modèle et récupère l'historique des erreurs
        errors_history = perceptron.fit(X, y, max_epochs=max_epochs)
        
        # Trace la courbe d'erreur pour ce taux d'apprentissage
        plt.plot(errors_history, label=f"Taux d'apprentissage = {lr}")
        
    plt.title("Convergence pour différents taux d'apprentissage")
    plt.xlabel("Époque")
    plt.ylabel("Nombre d'erreurs de classification")
    plt.legend()
    plt.show()

#J'ai choisi la seed 30 car elle permet de bien voir toutes les droites
np.random.seed(30)

#Données d'entrée 
X_and = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y_and = np.array([0, 0, 0, 1])

X_or = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y_or = np.array([0, 1, 1, 1])

X_xor = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y_xor = np.array([0, 1, 1, 0])

    
# Test des fonctions logiques
print("--- Tests sur les fonctions logiques ---")
perceptron_and = PerceptronSimple(0.1)
perceptron_and.fit(X_and, y_and)
print(f"Score AND: {perceptron_and.score(X_and, y_and)}")
plot_decision_boundary(perceptron_and, X_and, y_and, "Fonction AND")

perceptron_or = PerceptronSimple(0.1)
perceptron_or.fit(X_or, y_or)
print(f"Score OR: {perceptron_or.score(X_or, y_or)}")
plot_decision_boundary(perceptron_or, X_or, y_or, "Fonction OR")

perceptron_xor = PerceptronSimple(0.1)
perceptron_xor.fit(X_xor, y_xor) 
print(f"Score XOR: {perceptron_xor.score(X_xor, y_xor)}")
plot_decision_boundary(perceptron_xor, X_xor, y_xor, "Fonction XOR")

# Test sur données générées
print("\n--- Test sur les données générées linéairement séparables ---")
X_sep, y_sep = generer_donnees_separables(n_points=200, noise=0.3)
perceptron_sep = PerceptronSimple(learning_rate=0.1)
perceptron_sep.fit(X_sep, y_sep, max_epochs=10)
print(f"Score sur données séparables : {perceptron_sep.score(X_sep, y_sep)}")
plot_decision_boundary(perceptron_sep, X_sep, y_sep, "Données Séparables")


print("Analyse de la convergence")

# On définit une liste de taux d'apprentissage à tester
rates_to_test = [0.001, 0.01, 0.1, 1.0]

# On utilise les données séparables pour l'analyse
analyser_convergence(X_sep, y_sep, learning_rates=rates_to_test, max_epochs=15)