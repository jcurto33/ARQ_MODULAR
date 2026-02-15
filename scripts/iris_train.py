"""
scripts/train.py
─────────────────
Wrapper CLI para entrenar el modelo Iris.
"""

import sys
from pathlib import Path

# Añadir raíz del proyecto al path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.modules.iris_classifier.entrypoint import train


if __name__ == "__main__":
    print("═" * 50)
    print("  Entrenamiento — Iris Classifier")
    print("═" * 50)

    metrics = train()

    print(f"\n✅ Mejor modelo: {metrics['selected_model']}")
    for name, info in metrics["candidates"].items():
        print(f"   {name}: CV={info['cv_score']}  Test={info['test_accuracy']}")
