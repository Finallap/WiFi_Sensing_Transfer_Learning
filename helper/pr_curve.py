import torch
import torch.nn.functional as F


def create_pr_curve(writer, model, test_loader, csi_train_label):
    # helper function
    def add_pr_curve_tensorboard(class_index, test_probs, test_preds, global_step=0):
        '''
        Takes in a "class_index" from 0 to 9 and plots the corresponding
        precision-recall curve
        '''
        tensorboard_preds = test_preds == class_index
        tensorboard_probs = test_probs[:, class_index]

        writer.add_pr_curve(csi_train_label.categories[class_index],
                            tensorboard_preds,
                            tensorboard_probs,
                            global_step=global_step)
        writer.close()

    # 1. gets the probability predictions in a test_size x num_classes Tensor
    # 2. gets the preds in a test_size Tensor
    # takes ~10 seconds to run
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    class_probs = []
    class_preds = []
    with torch.no_grad():
        for batch_id, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)
            model.eval()
            output = model(data)
            class_probs_batch = [F.softmax(el, dim=0) for el in output]
            _, class_preds_batch = torch.max(output, 1)

            class_probs.append(class_probs_batch)
            class_preds.append(class_preds_batch)

    test_probs = torch.cat([torch.stack(batch) for batch in class_probs])
    test_preds = torch.cat(class_preds)

    # 绘制pr曲线到Tensor Board
    for i in range(len(csi_train_label.categories)):
        add_pr_curve_tensorboard(i, test_probs, test_preds)
