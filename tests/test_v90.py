from app.v90_core import OnlineFailureLearner, CONFIGS


def test_starts_neutral():
    l = OnlineFailureLearner()
    p, o, loss, a, level = l.predict((1, 1, 1, 2), 1, CONFIGS[1])
    assert p == 0.5 and o == 0 and loss == 0 and a == 1.0 and level == "global"


def test_update_is_after_trade_only():
    l = OnlineFailureLearner(); sig = (1, 1, 1, 2)
    l.update(sig, 1, True)
    p, o, loss, a, level = l.predict(sig, 1, CONFIGS[1])
    assert l.total_obs == 1 and l.total_losses == 1 and p > 0.5 and level == "global"


def test_aging():
    l = OnlineFailureLearner(max_age_days=5); sig = (1, 1, 1, 2)
    for d in (1, 2, 3, 4):
        l.update(sig, d, True)
    _, o, loss, _, level = l.predict(sig, 20, CONFIGS[1])
    assert o == 0 and loss == 0 and level == "global"


def test_suppression_requires_repeated_evidence():
    l = OnlineFailureLearner(); sig = (1, 1, 1, 2)
    for d in range(1, 12):
        l.update(sig, d, True)
    p, o, loss, a, _ = l.predict(sig, 12, CONFIGS[1])
    assert o >= 8 and loss >= 4 and p > 0.75 and a == 0.0
