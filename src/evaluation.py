from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score
from scipy import stats
import numpy as np

def holdout_evaluation(model, X, y, test_size=0.2):
    """Executa um simples Holdout na base de dados"""
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return accuracy_score(y_test, y_pred)

def repeated_holdout(model, X, y, test_size=0.2, n_splits=10):
    """Executa Holdout aleatório repetido"""
    scores = []
    for _ in range(n_splits):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=None)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        scores.append(accuracy_score(y_test, y_pred))
    return np.mean(scores), np.std(scores)

def ten_fold_holdout(model, X, y):
    """Executa 10x Holdout 50/50"""
    return repeated_holdout(model, X, y, test_size=0.5, n_splits=10)

def hypothesis_test(model1_scores, model2_scores):
    """Teste de hipótese para verificar diferença entre classificadores"""
    stat, p_value = stats.ttest_ind(model1_scores, model2_scores)
    return p_value < 0.05, p_value  # Retorna se há diferença significativa e o valor p

def confidence_interval_difference(model1_scores, model2_scores):
    """Calcula intervalo de confiança da diferença entre classificadores"""
    diff = np.array(model1_scores) - np.array(model2_scores)
    mean_diff = np.mean(diff)
    ci = stats.t.interval(0.95, len(diff)-1, loc=mean_diff, scale=stats.sem(diff))
    return ci

def overlap_confidence_intervals(model1_scores, model2_scores):
    """Verifica sobreposição de intervalos de confiança"""
    ci1 = stats.t.interval(0.95, len(model1_scores)-1, loc=np.mean(model1_scores), scale=stats.sem(model1_scores))
    ci2 = stats.t.interval(0.95, len(model2_scores)-1, loc=np.mean(model2_scores), scale=stats.sem(model2_scores))
    return not (ci1[1] < ci2[0] or ci2[1] < ci1[0])
