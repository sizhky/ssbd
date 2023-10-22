__version__ = "0.0.1"

from .cli import cli
from .preprocess import *
from .models import *

if __name__ == "__main__":
    cli()
