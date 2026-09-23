"""Ponto de entrada do construtor de planilhas."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from iso26315.editor import main

if __name__ == '__main__':
    sys.exit(main())
