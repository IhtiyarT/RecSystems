import numpy as np


class Metrics:
    
    def precision_at_k(self, recommended, relevant, k):
        recommended = recommended[:k]

        hits = len(set(recommended) & set(relevant))

        return hits / k
    
    def recall_at_k(self, recommended, relevant, k):
        recommended = recommended[:k]

        hits = len(set(recommended) & set(relevant))
        
        return hits / len(relevant)
    
    def idcg(self, recommended, relevant, k):
        recommended = recommended[:k]

        dcg = 0.0
        for i, item in enumerate(recommended):
            if item in relevant:
                dcg += 1 / np.log2(i + 2)
        
        idcg = 0.0
        for i in range(min(len(relevant), k)):
            idcg += 1 / np.log2(i + 2)

        if not idcg:
            return 0
        
        return dcg / idcg
    
    def rmse(self, y_true, y_pred):
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)

        return np.sqrt(
            np.mean((y_true - y_pred) ** 2)
            )