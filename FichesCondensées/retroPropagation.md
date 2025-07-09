La rétropropagation est une application directe et élégante de la **règle de dérivation en chaîne** du calcul différentiel, optimisée pour l'architecture des réseaux de neurones.

Son objectif est de calculer le gradient de la fonction de coût (Loss, notée L) par rapport à chaque paramètre (poids w et biais b) du réseau. Ce gradient, ∂L/∂w, nous indique comment un petit changement dans un poids w affecte l'erreur totale L.

### Notations et Contexte

Considérons un réseau de neurones avec K couches.

*   x : Le vecteur d'entrée.
    
*   y : La sortie attendue (la vérité terrain).
    
*   a⁽ᵏ⁾ : Le vecteur des activations (sorties) de la couche k. a⁽⁰⁾ = x.
    
*   z⁽ᵏ⁾ : Le vecteur des pré-activations (combinaisons linéaires) de la couche k.
    
*   w⁽ᵏ⁾ : La matrice des poids de la couche k (connecte la couche k-1 à la couche k).
    
*   b⁽ᵏ⁾ : Le vecteur des biais de la couche k.
    
*   f : La fonction d'activation.
    

Les équations de la **propagation avant (forward pass)** sont :

1.  z⁽ᵏ⁾ = w⁽ᵏ⁾ · a⁽ᵏ⁻¹⁾ + b⁽ᵏ⁾
    
2.  a⁽ᵏ⁾ = f(z⁽ᵏ⁾)
    

La sortie finale du réseau est ŷ = a⁽ᴷ⁾. La fonction de coût est L = C(ŷ, y), par exemple l'erreur quadratique moyenne L = ½(ŷ - y)².

### Le Cœur de l'Algorithme : Le "Delta" ou l'Erreur Propagée

L'élément central de la rétropropagation est le calcul d'une quantité δ⁽ᵏ⁾ pour chaque couche k. Cette quantité δ⁽ᵏ⁾ est définie comme le gradient de la fonction de coût L par rapport à la **pré-activation** z⁽ᵏ⁾ de cette couche :

δ⁽ᵏ⁾ = ∂L / ∂z⁽ᵏ⁾

Pourquoi z⁽ᵏ⁾ ? Parce que z⁽ᵏ⁾ est le point de jonction qui relie les poids et les biais de la couche k aux activations de la couche précédente. C'est le pivot idéal pour la règle de dérivation en chaîne.

### Les Quatre Équations Fondamentales de la Rétropropagation

L'algorithme repose sur quatre équations qui se calculent de la dernière couche vers la première.

**Équation 1 : L'erreur à la couche de sortie (δ⁽ᴷ⁾)**

On commence par calculer δ pour la dernière couche K. C'est le point de départ.

δ⁽ᴷ⁾ = ∂L/∂a⁽ᴷ⁾ ⊙ f'(z⁽ᴷ⁾)

*   ∂L/∂a⁽ᴷ⁾ est la dérivée de la fonction de coût par rapport à la sortie du réseau. Pour une erreur quadratique L = ½(a⁽ᴷ⁾ - y)², cette dérivée est simplement (a⁽ᴷ⁾ - y).
    
*   f'(z⁽ᴷ⁾) est la dérivée de la fonction d'activation évaluée au point de pré-activation z⁽ᴷ⁾.
    
*   ⊙ est le produit Hadamard (multiplication élément par élément).
    

**Cette équation traduit l'idée : "L'erreur à la sortie dépend de la rapidité à laquelle le coût change (∂L/∂a) ET de la sensibilité de la fonction d'activation (f')".**

**Équation 2 : L'erreur propagée aux couches cachées (δ⁽ᵏ⁾)**

C'est ici que la "rétro-propagation" a lieu. On calcule l'erreur δ pour une couche k en fonction de l'erreur de la couche suivante k+1.

δ⁽ᵏ⁾ = ((w⁽ᵏ⁺¹⁾)ᵀ · δ⁽ᵏ⁺¹⁾) ⊙ f'(z⁽ᵏ⁾)

*   (w⁽ᵏ⁺¹⁾)ᵀ est la transposée de la matrice de poids de la couche _suivante_. On l'utilise pour propager l'erreur "vers l'arrière".
    
*   δ⁽ᵏ⁺¹⁾ est l'erreur déjà calculée pour la couche suivante.
    
*   Le terme (w⁽ᵏ⁺¹⁾)ᵀ · δ⁽ᵏ⁺¹⁾ représente l'erreur de la couche k+1 pondérée et ramenée aux dimensions de la couche k.
    

**Cette équation est l'incarnation de la règle de dérivation en chaîne : "L'erreur dans la couche k est l'erreur de la couche k+1 propagée à travers les poids w⁽ᵏ⁺¹⁾, puis ajustée par la sensibilité de l'activation locale f'(z⁽ᵏ⁾)".**

**Équation 3 : Le gradient par rapport aux biais (∂L/∂b⁽ᵏ⁾)**

Une fois que nous avons δ⁽ᵏ⁾, le calcul des gradients pour les paramètres de cette couche devient direct.

∂L / ∂b⁽ᵏ⁾ = δ⁽ᵏ⁾

C'est l'équation la plus simple. Le gradient pour les biais est directement l'erreur δ de cette couche. Logique, car b⁽ᵏ⁾ n'influence L que via z⁽ᵏ⁾ de manière additive.

**Équation 4 : Le gradient par rapport aux poids (∂L/∂w⁽ᵏ⁾)**

∂L / ∂w⁽ᵏ⁾ = δ⁽ᵏ⁾ · (a⁽ᵏ⁻¹⁾)ᵀ

*   δ⁽ᵏ⁾ est l'erreur que nous venons de calculer.
    
*   (a⁽ᵏ⁻¹⁾)ᵀ est la transposée des activations de la couche _précédente_.
    
*   Le produit matriciel (ou produit externe si on traite un seul exemple) de ces deux termes nous donne la matrice de gradient pour tous les poids w⁽ᵏ⁾.
    

**Cette équation nous dit : "L'influence d'un poids wᵢⱼ⁽ᵏ⁾ sur l'erreur est proportionnelle à l'activation du neurone d'entrée aⱼ⁽ᵏ⁻¹⁾ (si l'entrée est forte, le poids a plus d'impact) ET à l'erreur du neurone de sortie δᵢ⁽ᵏ⁾". C'est l'implémentation de la règle de Hebb ("neurons that fire together, wire together").**

### L'Algorithme Complet

1.  **Forward Pass :**
    
    *   Pour chaque couche k de 1 à K, calculer z⁽ᵏ⁾ et a⁽ᵏ⁾. Stocker toutes ces valeurs.
        
2.  **Backward Pass :**
    
    *   **Initialisation :** Calculer δ⁽ᴷ⁾ pour la couche de sortie en utilisant l'**Équation 1**.
        
    *   **Propagation :** Pour chaque couche k de K-1 à 1 (en remontant) :
        
        *   Calculer δ⁽ᵏ⁾ en utilisant l'**Équation 2** (avec δ⁽ᵏ⁺¹⁾ et w⁽ᵏ⁺¹⁾ déjà connus).
            
    *   **Calcul des gradients :** Pour chaque couche k de K à 1 :
        
        *   Calculer ∂L/∂b⁽ᵏ⁾ avec l'**Équation 3**.
            
        *   Calculer ∂L/∂w⁽ᵏ⁾ avec l'**Équation 4**.
            
3.  **Mise à jour des paramètres :**
    
    *   Pour chaque couche k, mettre à jour les poids et les biais en utilisant une descente de gradient :
        
        *   w⁽ᵏ⁾ ← w⁽ᵏ⁾ - η · ∂L/∂w⁽ᵏ⁾
            
        *   b⁽ᵏ⁾ ← b⁽ᵏ⁾ - η · ∂L/∂b⁽ᵏ⁾(où η est le taux d'apprentissage).
            

La beauté de cet algorithme réside dans son efficacité : il évite les calculs redondants en réutilisant intelligemment les gradients calculés pour la couche k+1 afin de calculer ceux de la couche k.