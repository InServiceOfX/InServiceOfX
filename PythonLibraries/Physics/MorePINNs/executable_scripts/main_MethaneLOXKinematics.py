from pathlib import Path
import numpy as np
import sys

python_libraries_path = Path(__file__).resolve().parents[3]
print("\n ----- Python Libraries path:", python_libraries_path)
more_pinns_directory = \
    python_libraries_path / "Physics" / "MorePINNs"

if not str(more_pinns_directory) in sys.path:
    sys.path.append(str(more_pinns_directory))

from morepinns.MethaneLOXKinetics.MethaneLOXKinetics import (
    CombustionModel,
    create_training_data,
    train_model,
    plot_results)


if __name__ == "__main__":
    # Create model
    model = CombustionModel()
    
    # Create training data
    n_points = 1000
    X_train = create_training_data(n_points)
    
    # Train the model
    trained_model = train_model(model, X_train, epochs=10000)
    
    # Create evaluation points and plot results
    t_eval = np.linspace(0, 1, 200)