import sys
import torch
import dill._dill

try:
    import doclayout_yolo.nn.tasks
except ImportError:
    pass

# Fix torch serialization error by allowing the model class
safe_globals = [dill._dill._load_type]
if 'doclayout_yolo' in sys.modules:
     safe_globals.append(doclayout_yolo.nn.tasks.YOLOv10DetectionModel)

torch.serialization.add_safe_globals(safe_globals)

from mineru.cli.client import main

if __name__ == "__main__":
    sys.exit(main())
