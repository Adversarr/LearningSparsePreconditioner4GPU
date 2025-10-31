from pathlib import Path

def get_ncg_root():
    """Get the root directory of the neural_cg package."""
    return Path(__file__).parent.parent
