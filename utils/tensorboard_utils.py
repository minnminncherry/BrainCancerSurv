"""Optional TensorBoard logging, separate from model and loss implementations."""
from datetime import datetime
import math
import os
from pathlib import Path
from uuid import uuid4
import warnings


def start_tensorboard(args, fold, mutation_mode):
    """Give each experiment/fold/restart its own event directory."""
    try:
        from torch.utils.tensorboard import SummaryWriter
    except ImportError:
        warnings.warn('TensorBoard logging disabled: install tensorboard to enable it.', stacklevel=2)
        args._tensorboard_log_dir = None
        return
    default_root = Path(__file__).resolve().parents[1] / 'runs'
    root = Path(os.environ.get('TENSORBOARD_LOG_DIR') or default_root).expanduser().resolve()
    name = datetime.now().strftime('%Y%m%d_%H%M%S') + '_' + uuid4().hex[:8]
    path = root / args.modality / args.genomic_file_name / mutation_mode / f'fold_{fold}' / name
    args._tensorboard_log_dir = str(path)
    with SummaryWriter(log_dir=str(path)) as writer:
        writer.add_text('Run/model', str(args.modality), 0)
        writer.add_text('Run/dataset', str(args.genomic_file_name), 0)
        writer.add_text('Run/mutation_mode', mutation_mode, 0)
        writer.add_text('Run/loss', str(args.loss_func), 0)
    print(f'TensorBoard log: {path}')


def log_tensorboard(args, step, values):
    """Write scalars and close/flush immediately, including on errors."""
    path = getattr(args, '_tensorboard_log_dir', None)
    if not path:
        return
    from torch.utils.tensorboard import SummaryWriter
    with SummaryWriter(log_dir=path) as writer:
        for tag, value in values.items():
            if value is not None and math.isfinite(float(value)):
                writer.add_scalar(tag, float(value), int(step))
