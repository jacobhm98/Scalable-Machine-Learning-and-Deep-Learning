import torch.nn
from RegressionSBert import SBert as RegressionSBert
from ClassificationSBert import SBert as ClassificationSBert
import STSDataset
from SNLIDataset import SNLIDataset
from torch.utils.data import DataLoader
from scipy.stats import spearmanr


def main():
    model = train_classification_model()

def train_classification_model(model_file=None):
    if model_file is None:
        model = ClassificationSBert()
    else:
        model = torch.load(model_file)
    dataset = SNLIDataset()
    dataloader = DataLoader(dataset=dataset)

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-5)
    running_loss = 0
    for epoch in range(3):
        for i, data_point in enumerate(dataloader):
            features, label = data_point
            optimizer.zero_grad()
            outputs = model(features)
            try:
                loss = criterion(outputs, label)
            except:
                continue
            running_loss += loss.item()
            if i % 1000 == 999:
                print(i, running_loss)
                running_loss = 0
            loss.backward()
            optimizer.step()
        model.regression_objective(is_regression_objective=True)
        evaluate_model(model)
        model.regression_objective(is_regression_objective=False)
        torch.save(model, "classification_s_bert_" + str(epoch) +  ".model")
    return model


def train_regression_model(model_file=None):
    if model_file is None:
        model = RegressionSBert()
    else:
        model = torch.load(model_file)
    train_set = STSDataset.StsDataset('./stsbenchmark/sts-train.csv')
    data_loader = DataLoader(dataset=train_set, shuffle=True)

    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-5)
    model.train()
    running_loss = 0
    for epoch in range(1):
        for i, data_point in enumerate(data_loader):
            features, target = data_point
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, target)
            running_loss += loss.item()
            if i % 100 == 99:
                print(i, running_loss)
                running_loss = 0
            loss.backward()
            optimizer.step()
        torch.save(model, "regression_s_bert.model")
    return model

def evaluate_model(model):
    test_set = STSDataset.StsDataset('./stsbenchmark/sts-test.csv')
    test_data_loader = DataLoader(dataset=test_set)
    model.eval()
    predicted = []
    real_values = []
    for data_point in test_data_loader:
        features, target = data_point
        outputs = model(features)
        predicted.append(outputs.item())
        real_values.append(target.item())
    spearman_coefficient, p_val = spearmanr(predicted, real_values)
    print("Observed correlation", spearman_coefficient)


main()
