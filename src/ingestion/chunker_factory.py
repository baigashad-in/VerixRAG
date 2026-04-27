from chunkers import FixedSizeChunker, RecursiveChunker, SemanticChunker

class ChunkerFactory:
    """Swap chunking strategies without changing any ohter code.
    Built an evaluation pipeline. Test multiple strategies, and pick the one with the best retrieval metrics. This factory makes that experimentation possible.
    """

    strategies = {
        "fixed": FixedSizeChunker,
        "recursive": RecursiveChunker,
        "semantic": SemanticChunker,
    }


    @classmethod
    def create(cls, strategy: str, **kwargs):
        if strategy not in cls.strategies:
            available = ", ".join(cls.strategies.keys())
            raise ValueError(
                f"Unknown strategy '{strategy}'. Choose from: {available}"
            )
        return cls.strategies[strategy](**kwargs)

