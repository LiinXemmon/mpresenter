from pathlib import Path
import sys


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent
    sys.path.insert(0, str(project_root / "src"))

    from mpresenter.cli import main

    main()
