import re


def extract_confusion_matrices(log_file):
    folds = log_file.split('Fold ')
    confusion_matrices = {}
    for log in folds:
        matches = re.findall(r'Confusion matrices:(.*?)}', log, re.DOTALL)
        for match in matches:
            match = match.replace('{', '').replace('}', '')
            matrices = match.split('), ')
            matrices[-1] = matrices[-1].replace(')', '')
            for matrix in matrices:
                key, value = matrix.split(': array(')
                confusion_matrices[eval(key)] = eval(value)
    return confusion_matrices


def extract_loss_history(log_file):
    folds = log_file.split('Fold ')
    loss_history = {}
    for log in folds:
        if len(log) <= 0:
            continue
        key = eval(log[0])
        loss_history[key] = []
        losses = re.findall(r'Loss: ([\d.]*)', log)
        for loss in losses:
            loss = loss.replace('Loss: ', '')
            loss_history[key].append(float(loss))

    return loss_history


def extract_val_history(log_file):
    folds = log_file.split('Fold ')
    loss_history = {}
    for log in folds:
        if len(log) <= 0:
            continue
        key = eval(log[0])
        loss_history[key] = []
        losses = re.findall(r'Val acc: ([\d.]*)', log)
        for loss in losses:
            loss = loss.replace('Val acc: ', '')
            loss_history[key].append(float(loss))

    return loss_history
