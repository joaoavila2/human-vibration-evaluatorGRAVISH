"""Ponto de entrada do aplicativo."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from iso26315.gui import main

if __name__ == '__main__':
    sys.exit(main())
