from spam_classifier import predict, train


def test_model_runs_and_predicts():
    model, accuracy = train()
    assert 0.0 <= accuracy <= 1.0
    assert predict(model, "Win a free prize now, click here") == "spam"
    assert predict(model, "Can you send me the homework notes?") == "ham"
