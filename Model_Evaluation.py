import matplotlib
import matplotlib.pyplot as plt
import numpy as np
n_folds = 3
ds_name = [["AD","NC"]]



plot_history_path="/coronal_scores"
#plot_history_path="/sagital_scores"
#plot_history_path="/axial_scores"


plot_data_type= ["ROC_AUC", "evaluation","training"]

def get_plot_data(history_path, prefix, plot_data_type,columns=1):
    whole_history_list = list()
    whole_data_list = list()
    history_list = list()
    data_list = list()
    history_txt_prefix="{}//{}_history_{}_vs_{}".format(str(history_path),str(plot_data_type),str(prefix[0]),str(prefix[1]))
    for j in range(1,n_folds+1):
        history_txt="{}_fold_index_{}.txt".format(str(history_txt_prefix),str(j))
        history_list.append(history_txt)
        data = np.loadtxt(history_txt, delimiter=',', usecols = columns)
        data_list.append(data)

    whole_history_list.append(history_list)
    whole_data_list.append(data_list)
    return np.asanyarray(whole_data_list)

def plot_ROC_AUC(plot_history_path,prefix,plot_data_type):
    mean = list()
    std = list()
    ROC_AUC = get_plot_data(plot_history_path, prefix, plot_data_type)
    #print(ROC_AUC.shape)
    #for i in range(ROC_AUC.shape[-1]):
    for i in range(60):
        mean.append(np.mean(ROC_AUC[:, :, i]))
        std.append(np.std(ROC_AUC[:, :, i]))
    mean = np.asanyarray(mean)
    std = np.asanyarray(std)
    plt.plot(mean)
    plt.fill_between(list(range(len(mean))), mean + std, mean - std, color='gray', alpha=0.2)
    #plt.title('AD vs NC classification ROC AUC')
    plt.title('{} vs {} classification ROC AUC axial_classifier'.format(str(prefix[0]), str(prefix[1])))
    plt.ylabel('ROC AUC')
    plt.xlabel('epoch')
    plt.legend(['Validation ROC AUC mean', 'Validation ROC AUC std'], loc='lower left')
    plt.show()

def calculate_results(results):
    mean = list()
    std = list()
    mean.append(np.mean(results[:,:,-1 ]))
    std.append(np.std(results[:, : ,-1]))
    mean = np.asanyarray(mean)
    std = np.asanyarray(std)
    return mean, std

def print_evaluation_results(plot_data_type):
    mean_list = list()
    std_list = list()
    results = get_plot_data(plot_history_path, ds_name[0], plot_data_type)
    mean, std = calculate_results(results)
    mean_list.append(mean)
    std_list.append(std)
    mean_list = np.asanyarray(mean_list)
    std_list = np.asanyarray(std_list)
    if plot_data_type == "ROC_AUC" :
        title = plot_data_type
    else :
        title = "{} acc".format(str(plot_data_type))
    print("\n    {} : [mean] ± [std]\n".format(str(title)))
    for i in range(len(ds_name)):
        print("{} vs {} : {} ± {}".format(str(ds_name[i][0]),str(ds_name[i][1]),mean_list[i],std_list[i]))
def training_acc_loss(plot_history_path, prefix, plot_data_type):
    mean_loss = list()
    mean_acc=list()
    std_loss = list()
    std_acc = list()
    train_loss = get_plot_data(plot_history_path, prefix, plot_data_type,1)
    train_acc= get_plot_data(plot_history_path, prefix, plot_data_type,columns=2)
    #print(ROC_AUC.shape)
    #for i in range(ROC_AUC.shape[-1]):
    for i in range(60):
        mean_loss.append(np.mean(train_loss[:, :, i]))
        mean_acc.append(np.mean(train_acc[:, :, i]))
        std_loss.append(np.std(train_loss[:, :, i]))
        std_acc.append(np.std(train_acc[:, :, i]))
    mean_loss = np.asanyarray(mean_loss)
    mean_acc = np.asanyarray(mean_acc)
    std_loss = np.asanyarray(std_loss)
    std_acc = np.asanyarray(std_acc)
    plt.plot(mean_loss)
    plt.plot(mean_acc)
    plt.fill_between(list(range(len(mean_loss))), mean_loss + std_loss, mean_loss - std_loss, color='gray', alpha=0.2)
    plt.fill_between(list(range(len(mean_acc))), mean_acc+ std_acc, mean_acc - std_acc, color='gray', alpha=0.2)
    #plt.title('AD vs NC training loss acc')
    plt.title('{} vs {} training loss vs acc'.format(str(prefix[0]), str(prefix[1])))
    plt.ylabel('acc vs loss')
    plt.xlabel('epoch')
    plt.legend(['Training acc  mean', 'Training loss mean'], loc='lower left')
    plt.show()
def evaluation_acc_loss(plot_history_path, prefix, plot_data_type):
    mean_loss = list()
    mean_acc=list()
    std_loss = list()
    std_acc = list()
    train_loss = get_plot_data(plot_history_path, prefix, plot_data_type,2)
    train_acc= get_plot_data(plot_history_path, prefix, plot_data_type,columns=1)
    #print(ROC_AUC.shape)
    #for i in range(ROC_AUC.shape[-1]):
    for i in range(60):
        mean_loss.append(np.mean(train_loss[:, :, i]))
        mean_acc.append(np.mean(train_acc[:, :, i]))
        std_loss.append(np.std(train_loss[:, :, i]))
        std_acc.append(np.std(train_acc[:, :, i]))
    mean_loss = np.asanyarray(mean_loss)
    mean_acc = np.asanyarray(mean_acc)
    std_loss = np.asanyarray(std_loss)
    std_acc = np.asanyarray(std_acc)
    plt.plot(mean_loss)
    plt.plot(mean_acc)
    plt.fill_between(list(range(len(mean_loss))), mean_loss + std_loss, mean_loss - std_loss, color='gray', alpha=0.2)
    plt.fill_between(list(range(len(mean_acc))), mean_acc+ std_acc, mean_acc - std_acc, color='gray', alpha=0.2)
    #plt.title('AD vs NC training loss acc')
    plt.title('{} vs {} evaluation loss vs acc'.format(str(prefix[0]), str(prefix[1])))
    plt.ylabel('acc vs loss')
    plt.xlabel('epoch')
    plt.legend(['Evaluation acc  mean', 'Evaluation loss mean'], loc='lower left')
    plt.show()



plot_ROC_AUC(plot_history_path, ds_name[0], plot_data_type[0])
print_evaluation_results(plot_data_type[0])
print_evaluation_results(plot_data_type[1])
evaluation_acc_loss(plot_history_path, ds_name[0], plot_data_type[2])
training_acc_loss(plot_history_path, ds_name[0], plot_data_type[1])