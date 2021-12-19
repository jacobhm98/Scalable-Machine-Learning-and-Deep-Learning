import torch.nn

from RegressionSBert import SBert
import stsdataset
from torch.utils.data import DataLoader
from scipy.stats import spearmanr


def main():
    #model = SBert()
    model = torch.load('regression_s_bert.model')
    train_set = stsdataset.StsDataset('./stsbenchmark/sts-train.csv')
    train_data_loader = DataLoader(dataset=train_set, shuffle=True)
    test_set = stsdataset.StsDataset('./stsbenchmark/sts-test.csv')
    test_data_loader = DataLoader(dataset=test_set)
    train_model(model, train_data_loader)
    evaluate_model(model, test_data_loader)


def train_model(model, data_loader):
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-5)
    model.train()
    running_loss = 0
    for epoch in range(3):
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

def evaluate_model(model, test_set):
    model.eval()
    predicted = []
    real_values = []
    for data_point in test_set:
        features, target = data_point
        outputs = model(features)
        predicted.append(outputs.item())
        real_values.append(target.item())
    spearman_coefficient, p_val = spearmanr(predicted, real_values)
    print("Observed correlation", spearman_coefficient)


main()
