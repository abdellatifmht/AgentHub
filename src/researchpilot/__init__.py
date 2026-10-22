def main() -> None:
    topic = input("Quel sujet veux-tu étudier ? ")
    if not topic.strip():
        print("Erreur : le sujet ne peut pas être vide.")
        return
    print(f"\nAgentHub démarre une recherche sur : {topic}")
    print("Étape 1 : sujet reçu.")
    print("Étape 2 : recherche Web à implémenter.")
    print("Étape 3 : analyse à implémenter.")
    print("Étape 4 : génération du rapport à implémenter.")
