import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import numpy as np

def plot_loss_lr_batch(model_dict):
    new = dict()
    for j in model_dict.keys():
        name = j.split(',')
        l = name[0][1:]
        b = name[1][:-1]
        if b not in new.keys():
            new[b] = [[l], [model_dict[j][0].history['loss'][-1]]]
        else:
            new[b][0].append(l)
            new[b][1].append(model_dict[j][0].history['loss'][-1])

    plt.figure(figsize=(6,4))
    for j in new.keys():
        plt.plot(new[j][0],new[j][1])

    plt.title('Loss for each batch size')
    plt.xlabel('Learning Rate')
    plt.ylabel('Loss')
    plt.legend(new.keys())
    
def plot_results_1(model_info_list, name):
    fig = plt.figure(figsize=(15, 6))
    ax1 = fig.add_subplot(121)
    symbols = [':','-','--']
    for i,model_info in enumerate(model_info_list):
        n = len(model_info.history['acc'])
        ax1.plot(range(1, n+1), model_info.history['acc'], 
                 linewidth=3, ls=symbols[i], color='b')
        ax1.plot(range(1, n+1), model_info.history['val_acc'], 
                 linewidth=3, ls=symbols[i], color ='orange')
        plt.ylabel('Accuracy', fontsize=20)
        plt.xlabel('Epoch', fontsize=20)
        plt.xlim(0,80)
        ax1.set_xticks(np.arange(1,n+1), n/10)
        ax1.legend(['train', 'val'], loc='best', fontsize=15)
        ax1.text(60,0.5 - 0.1 *i,name[i]+' '+symbols[i],fontsize=20)
    ax2 = fig.add_subplot(122)

    for i,model_info in enumerate(model_info_list):
        n = len(model_info.history['acc'])
        ax2.plot(range(1, n+1), model_info.history['loss'],
                 linewidth=3, ls=symbols[i], color='b')
        ax2.plot(range(1, n+1), model_info.history['val_loss'], 
                 linewidth=3, ls=symbols[i], color='orange')
        plt.ylabel('Loss', fontsize = 20)
        plt.xlabel('Epoch', fontsize = 20)
        plt.xlim(0,80)
        ax2.set_xticks(np.arange(1,n+1), n/10)

    plt.show()
        
    
def plot_confusion(test_list, pred, abbreviation):
    fig, ax = plt.subplots(1)
    confusion = confusion_matrix(test_list, pred)
    ax = sns.heatmap(confusion, ax = ax, cmap=plt.cm.Oranges, annot=True)
    ax.set_xticklabels(abbreviation)
    ax.set_yticklabels(abbreviation)
    plt.title('Confusion Matrix', size=20)
    plt.ylabel('True', size=16)
    plt.xlabel('Predicted', size=16)
    plt.show();
