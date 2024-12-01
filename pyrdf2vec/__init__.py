import nest_asyncio

from .rdf2vec import RDF2VecTransformer
from GNCE import PROJECT_PATH

# bypass the asyncio.run error for the Notebooks.
nest_asyncio.apply()

__all__ = [
    "RDF2VecTransformer",
]
__version__ = "0.2.3"

WALK_PATH = PROJECT_PATH + "/walks"
