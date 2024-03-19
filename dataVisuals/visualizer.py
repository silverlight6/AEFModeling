import matplotlib.pyplot as plt

def show_variable_comparison(xLabel, yLabel, df, principle_stats):
    plt.figure(figsize=(10, 10))
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=14)
    plt.xlabel(xLabel, fontsize=20)
    plt.ylabel(yLabel, fontsize=20)
    plt.title("Principal Component Analysis of sensor values", fontsize=20)
    # targets = ["productive", "nan", "distracted"]
    targets = ["non-transition", "transition", "nan"]
    colors = ['g', 'r', 'b']
    for target, color in zip(targets, colors):
        # All indicies for the target in question
        indiciesToKeep = df['focus'] == target
        plt.scatter(principle_stats.loc[indiciesToKeep, xLabel],
                    principle_stats.loc[indiciesToKeep, yLabel], c=color, s=1)
    plt.legend(targets, prop={'size': 15})
    plt.show()


def show_confustion_matrix(cm):
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.matshow(cm, cmap=plt.cm.Blues, alpha=0.3)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(x=j, y=i, s=cm[i, j], va='center', ha='center', size='xx-large')

    plt.xlabel('Predictions', fontsize=16)
    plt.ylabel('Actuals', fontsize=16)
    plt.title('Confusion Matrix', fontsize=16)
    plt.show()


def analyze_model_features(model, dataset):
    features = zip(list(model.feature_importances_), list(dataset.columns))
    feature_list = sorted(list(features), key=lambda a: a[0], reverse=True)
    plt.figure(figsize=(10, 4))

    # creating the bar plot
    plt.bar([f[1] for f in feature_list],
            [f[0] for f in feature_list],
            color='limegreen',
            width=0.4)
    plt.subplots_adjust(bottom=0.4)

    plt.xlabel("Feature")
    plt.ylabel("Score")
    plt.title("Model Features")

    plt.xticks(
        rotation=45,
        horizontalalignment='right',
        fontweight='light',
    )

    plt.show()
