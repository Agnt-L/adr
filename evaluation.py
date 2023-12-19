import numpy as np
from sklearn.metrics import f1_score, accuracy_score, matthews_corrcoef, precision_recall_curve, auc, confusion_matrix, \
    roc_curve, mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.use('Qt5Agg')

# Final evaluation for binary model. Saves evaluation results for f1-score, accuracy, mcc, pr-auc-score
# and confusion matrix. Draws and saves PR and ROC curves.
def binary_evaluation_final(y_true, score, filename, model_name, thresh):
    # metrics to calculate
    scorers = {
        'f1_score': f1_score,
        'acc': accuracy_score,
        'mcc': matthews_corrcoef
    }

    # get precision-recall curve and pr-auc-score
    precision, recall, thresholds = precision_recall_curve(y_true, score)
    pr_auc = auc(recall, precision)
    # get roc-curve and roc-auc-score
    fpr, tpr, threshs = roc_curve(y_true, score)
    roc_auc = auc(fpr, tpr)

    metrics_score = []
    # append calculated metrics for score i
    j = 0
    for scorer in scorers:
        y_pred = [x > thresh[j] for x in score]
        metrics_score.append(scorers[scorer](y_true, y_pred))
        j += 1

    # save results to evaluation file
    with open('data/' + filename, 'w') as f:
        f.write(f'pr-auc-score: {pr_auc}\n')
        f.write(f'roc-auc-score: {roc_auc}\n')
        f.write('\n')

        j = 0
        for scorer in scorers:
            f.write(f'{scorer}: {metrics_score[j]}\n')
            f.write(f'best threshold: {thresh[j]}\n')
            f.write('best threshold - conf matrix:\n')
            f.write(str(confusion_matrix(y_true, [x > thresh[j] for x in score])) + '\n')
            f.write('\n')
            j += 1

    # save roc-curve
    fig2 = plt.figure()
    plt.plot(fpr, tpr)
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.savefig('data/' + filename + '_roc_curve')
    plt.close(fig2)

    plot_pr_curve(y_true, score, model_name, filename)

# Final evaluation for binary model. Predicted values are given directly and not as scores.
def binary_evaluation_final_pred(y_true, y_pred, filename):
    # metrics to calculate
    scorers = {
        'f1_score': f1_score,
        'acc': accuracy_score,
        'mcc': matthews_corrcoef
    }

    metrics_score = []
    j = 0
    # iterate over metrics
    for scorer in scorers:
        metrics_score.append(scorers[scorer](y_true, y_pred))
        j += 1

    # save results to evaluation file
    with open('data/' + filename, 'w') as f:
        j = 0
        for scorer in scorers:
            f.write(f'{scorer}: {metrics_score[j]}\n')
            # calculate and print corresponding confusion matrix
            f.write('conf matrix:\n')
            f.write(str(confusion_matrix(y_true, y_pred)) + '\n')
            f.write('\n')
            j += 1

# Evaluation of regression model. Saves RMSE, MSE and MAE in file data/regression_results.txt
def regression_evaluation(y_pred, y_true):
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = mean_squared_error(y_true=y_true, y_pred=y_pred, squared=False)

    with open('data/regression_results.txt', 'w') as f:
        f.write(f'rmse: {rmse}, standardized rmse: {rmse / np.std(y_true)}\n')
        f.write(f'mse: {mse}, standardized mse: {mse / np.std(y_true)}\n')
        f.write(f'mae: {mae}, standardized mae: {mae / np.std(y_true)}\n')

# Plots PR curve of given predicted scores [model_probs] and true values [test_y]. Also draws the amount of
# positive samples in the data as a horizontal baseline. The model curve is labeles as [model_name].
def plot_pr_curve(test_y, model_probs, model_name, filename):
    # calculate the no skill line as the proportion of the positive class
    no_skill = test_y.count(True) / len(test_y)
    # plot the no skill precision-recall curve
    plt.plot([0, 1], [no_skill, no_skill], linestyle='--', label='No Skill')
    # plot model precision-recall curve
    precision, recall, _ = precision_recall_curve(test_y, model_probs)
    plt.plot(recall, precision, marker='.', label=model_name)
    # axis labels
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    # show the legend
    plt.legend()
    # save the plot
    plt.savefig('data/' + filename + '_pr_curve')
    plt.close()

