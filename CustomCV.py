import numpy as np

class CustomCV():
    @classmethod
    def chunk_indexes(cls, indexes, chunk = 10):
        np.random.shuffle(indexes)
        chunked_data = np.array_split(indexes, chunk)
        return np.array(chunked_data)
    
    @classmethod
    def split(cls, x, y, k=10):
        train_data = np.array(x)
        indexes = np.array([x for x in range(len(y))])
        folds_indexes = cls.chunk_indexes(indexes, k) 
        folds = []
        for fold in folds_indexes:
            test = fold
            train = np.array([x for x in indexes if x not in fold])
            folds.append((train, test))
            
        return folds
