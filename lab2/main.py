import torch.nn
from RegressionSBert import SBert as RegressionSBert
from ClassificationSBert import SBert as ClassificationSBert
import STSDataset
from SNLIDataset import SNLIDataset
from torch.utils.data import DataLoader
from scipy.stats import spearmanr


def main():
    #model = train_regression_model(model_file="./regression_s_bert_2.model")
    model = train_classification_model()

def train_classification_model(model_file=None):
    if model_file is None:
        model = ClassificationSBert()
    else:
        model = torch.load(model_file)
    dataset = SNLIDataset()
    dataloader = DataLoader(dataset=dataset)
    model.train()

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-5)
    running_loss = 0
    for epoch in range(1):
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
            if i % 10000 == 9999:
                model.regression_objective(is_regression_objective=True)
                evaluate_model(model)
                model.regression_objective(is_regression_objective=False)
                model.train()
                torch.save(model, "classification_s_bert_" + str(i) + ".model")
            loss.backward()
            optimizer.step()
    return model


def train_regression_model(model_file=None):
    if model_file is None:
        model = RegressionSBert()
    else:
        model = torch.load(model_file)
        model.regression_objective(is_regression_objective=True)
    train_set = STSDataset.StsDataset('./stsbenchmark/sts-train.csv')
    data_loader = DataLoader(dataset=train_set, shuffle=True)

    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters())
    running_loss = 0
    for epoch in range(3):
        model.train()
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
        evaluate_model(model)
        torch.save(model, "regression_s_bert_" + str(3 + epoch) + ".model")
    return model

def evaluate_model(model):
    test_set = STSDataset.StsDataset('./stsbenchmark/sts-test.csv')
    test_data_loader = DataLoader(dataset=test_set)
    model.eval()
    model.requires_grad_(False)
    predicted = []
    real_values = []
    for data_point in test_data_loader:
        features, target = data_point
        outputs = model(features)
        predicted.append(outputs.item())
        real_values.append(target.item())
    spearman_coefficient, p_val = spearmanr(predicted, real_values)
    model.requires_grad_(True)
    print("Observed correlation", spearman_coefficient)


main()
