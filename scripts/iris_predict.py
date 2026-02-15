"""
scripts/predict.py
───────────────────
Wrapper CLI para predicción con el modelo Iris.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.modules.iris_classifier.entrypoint import predict


if __name__ == "__main__":
    # Ejemplo hardcoded — en un caso real se leería de un archivo o argumentos
    examples = [
        [5.1, 3.5, 1.4, 0.2],  # setosa típica
        [6.7, 3.0, 5.2, 2.3],  # virginica típica
        [5.9, 3.0, 4.2, 1.5],  # versicolor típica
    ]

    print("═" * 50)
    print("  Predicción — Iris Classifier")
    print("═" * 50)

    for features in examples:
        result = predict(features)
        print(f"  {features} → {result['prediction']}  {result['probabilities']}")
