import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

def plot_KNN_ROC(closed_min_dist, open_min_dist):
    plt.rcParams["figure.figsize"] = (8,8)
    plt.rcParams.update({'font.size': 22})
    yt = [1 for elem in open_min_dist] + [0 for elem in closed_min_dist]
    ys = open_min_dist + closed_min_dist
    
    fpr, tpr, _ = roc_curve(yt, ys)
    label_str = f'(AUROC: {auc(fpr, tpr):.3f})'
    plt.plot(fpr, tpr, label=label_str)

    plt.plot([0, 1], [0, 1], color="navy", linestyle="--")
    plt.xlim([-0.01, 1.0])
    plt.ylim([-0.01, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(loc='lower right')
    plt.title('Closed/Open Classification ROC (NN Min Distance)')
    plt.show()
    return auc(fpr, tpr)
