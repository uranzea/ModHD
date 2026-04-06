import sys
from pathlib import Path

# Ensure the package is importable when running this script directly
sys.path.append(str(Path(__file__).resolve().parents[1]))

from dashboard_app import main

if __name__ == "__main__":
    main()
