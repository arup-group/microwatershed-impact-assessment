import os
import pytest
from app import get_whitebox_binary_path  # Adjust if your function is in a different module

def test_whitebox_path_exists_or_errors():
    try:
        path = get_whitebox_binary_path()
        assert os.path.exists(path), "WhiteboxTools binary path does not exist"
    except RuntimeError as e:
        assert "Unsupported OS" in str(e)
    except FileNotFoundError as e:
        assert "WhiteboxTools binary not found" in str(e)
