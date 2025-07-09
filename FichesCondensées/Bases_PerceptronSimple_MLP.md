### **Fiche de Révision Express - Perceptron & Réseaux de Neurones**

#### **1\. Le Concept Fondamental : Le Perceptron Simple (TP 1)**

C'est le **plus petit bloc de construction**, un neurone artificiel unique.

*   **Son but :** Classifieur binaire. Il apprend à tracer **une seule droite** (un hyperplan en dimensions > 2) pour séparer deux groupes de données.
    
*   **Sa limite :** Il ne fonctionne QUE si les données sont **linéairement séparables**. Il est incapable de résoudre le problème **XOR**.
    
*   **La Formule Essentielle du Perceptron :**Le calcul se fait en deux temps :
    
    1.  Generated codez = (w₁\*x₁ + w₂\*x₂ + ... + wₙ\*xₙ) + bUse code [with caution](https://support.google.com/legal/answer/13505487).Ou en notation vectorielle (à connaître !) :z = w · x + b
        
    2.  **Activation ŷ :** On passe z dans une fonction d'activation pour obtenir la prédiction finale.ŷ = f(z)
        
    
    *   x: Vecteur des données d'entrée (les "features").
        
    *   w: Vecteur des poids (l'importance de chaque feature). **C'est ce que le modèle apprend !**
        
    *   b: Le biais (permet de décaler la droite de séparation). **Lui aussi est appris.**
        
    *   z: Le score brut, avant la décision.
        
    *   f: La fonction d'activation.
        
    *   ŷ: La prédiction finale (ex: 0 ou 1, -1 ou 1).
        

#### **2\. Le Cœur du Neurone : La Fonction d'Activation (TP 1)**

*   **À quoi ça sert ?** C'est une fonction mathématique qui décide si un neurone doit "s'activer" (transmettre un signal) ou non, en se basant sur le score z.
    
*   **Pourquoi c'est CRUCIAL ?** Elle introduit de la **non-linéarité**. Sans elle, empiler des couches de neurones reviendrait à faire une seule grosse transformation linéaire, et le réseau ne pourrait pas apprendre de formes complexes (comme pour le XOR).
    
*   **Les Formules des Fonctions d'Activation à Connaître :**
    
    *   **Heaviside (Fonction échelon) :** L'ancêtre. Binaire, tout ou rien.f(z) = 1 si z ≥ 0, sinon 0_Problème :_ Sa dérivée est nulle partout sauf en zéro (où elle est infinie), ce qui rend la descente de gradient impossible.
        
    *   **Sigmoïde (Fonction logistique) :** Écrase les valeurs entre 0 et 1.f(z) = 1 / (1 + e⁻ᶻ)_Usage :_ Idéale pour la couche de sortie d'un problème de classification binaire (interprétable comme une probabilité).
        
    *   **Tangente Hyperbolique (Tanh) :** Écrase les valeurs entre -1 et 1.f(z) = (eᶻ - e⁻ᶻ) / (eᶻ + e⁻ᶻ)_Usage :_ Souvent préférée à la sigmoïde dans les couches cachées car sa sortie est centrée sur 0, ce qui peut accélérer la convergence.
        
    *   **ReLU (Rectified Linear Unit) :** La plus populaire aujourd'hui.f(z) = max(0, z)_Usage :_ Très rapide à calculer, elle est le standard dans les couches cachées des réseaux profonds. Évite le problème de "disparition du gradient" (vanishing gradient) pour les valeurs positives.
        
    *   **Leaky ReLU :** Une amélioration de ReLU.f(z) = z si z > 0, sinon α\*z (avec α petit, ex: 0.01)_Usage :_ Empêche les neurones de "mourir" complètement pour les valeurs négatives en laissant passer un petit gradient.
        

#### **3\. La Propagation Avant (Forward Pass) : Faire une Prédiction (TP 1 & 2)**

C'est le processus qui permet au réseau de **calculer une prédiction** à partir d'une entrée. L'information ne va que dans un sens : de l'entrée vers la sortie.

*   **Pour un Perceptron Simple :** C'est juste l'application de la formule vue plus haut.
    
    1.  Prendre les entrées x.
        
    2.  Calculer z = w · x + b.
        
    3.  Calculer ŷ = f(z).
        
    4.  C'est fini, ŷ est la prédiction.
        
*   **Pour un Réseau Multicouches (MLP) :** C'est une **cascade** de propagations avant. La sortie d'une couche devient l'entrée de la couche suivante.
    
    1.  **Entrée -> Couche Cachée 1 :** On prend x et on calcule la sortie de la première couche a¹ pour _tous ses neurones_.
        
    2.  **Couche Cachée 1 -> Couche Cachée 2 :** On prend a¹ comme nouvelle entrée et on calcule la sortie de la deuxième couche a².
        
    3.  **... (on répète pour toutes les couches cachées)**
        
    4.  **Dernière Couche Cachée -> Couche de Sortie :** On prend la sortie de la dernière couche cachée et on calcule la prédiction finale ŷ.
        

En résumé, la propagation avant, c'est simplement **"traverser le réseau de gauche à droite"** pour obtenir un résultat.

#### **4\. Les Pièges de l'Apprentissage : Underfitting vs Overfitting (TP 2)**

L'objectif de l'entraînement n'est pas d'être parfait sur les données d'entraînement, mais d'être bon sur des **données inconnues** (généralisation).

*   **Underfitting (Sous-apprentissage) : "Le modèle est trop bête"**
    
    *   **Symptômes :** Le modèle est **mauvais partout**, sur les données d'entraînement comme sur les données de test. La courbe d'erreur reste élevée.
        
    *   **Analogie :** Un étudiant qui a si peu révisé qu'il rate son partiel blanc (entraînement) ET le vrai partiel (test).
        
    *   **Cause :** Le modèle est trop simple pour capturer la complexité des données (ex: utiliser un Perceptron simple pour le problème XOR).
        
    *   **Solution :** Utiliser un modèle plus complexe (plus de neurones, plus de couches).
        
*   **Overfitting (Sur-apprentissage) : "Le modèle a appris par cœur"**
    
    *   **Symptômes :** Le modèle est **excellent sur les données d'entraînement** (accuracy proche de 100%) mais **mauvais sur les nouvelles données** (test/validation). La courbe d'erreur d'entraînement descend très bas, tandis que celle de validation stagne ou remonte.
        
    *   **Analogie :** Un étudiant qui a mémorisé les solutions des exercices du livre par cœur, mais qui est incapable de résoudre un nouvel exercice qui utilise la même logique.
        
    *   **Cause :** Le modèle est trop complexe pour la quantité de données disponibles. Il a appris le "bruit" et les cas particuliers des données d'entraînement au lieu de la tendance générale.
        
    *   **Solutions :**
        
        1.  **Avoir plus de données** (la meilleure solution, mais pas toujours possible).
            
        2.  Utiliser un modèle plus simple.
            
        3.  Utiliser des techniques de **régularisation** (non vues en détail mais bon à savoir : Dropout, L1/L2).
            
        4.  **Early Stopping** (arrêter l'entraînement quand les performances sur le set de validation commencent à se dégrader).
            

**Si les notions sont bien maitrisées :**

*   Je peux expliquer la différence entre un problème **linéairement séparable** (AND) et un problème qui ne l'est pas (XOR).
    
*   Je connais par cœur la formule du perceptron (z = w·x + b et ŷ = f(z)).
    
*   Je sais ce qu'est une fonction d'activation, **pourquoi elle est non-linéaire** et je peux citer/écrire les formules de ReLU, Sigmoïde et Tanh.
    
*   Je peux décrire le processus de **propagation avant (forward pass)** pour un MLP comme une succession de calculs de couche en couche.
    
*   Je peux définir **l'overfitting** et **l'underfitting** et expliquer comment les détecter (en comparant les performances train/test).
    
*   Je sais que la **rétropropagation (backward pass)** est l'algorithme qui permet d'ajuster les poids en propageant l'erreur de la sortie vers l'entrée.