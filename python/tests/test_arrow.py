from pathlib import Path

import pmmlruntime as pm


def test_run_rejects_arrow_with_clear_error():
    root = Path(__file__).resolve().parents[2]
    sess = pm.InferenceSession(str(root / "bench/pmml/DecisionTreeIris.pmml"))
    try:
        sess.run(None, "not-a-dict")
        assert False, "should raise TypeError"
    except TypeError as e:
        assert "dict or list[dict]" in str(e)
        assert "to_pylist" in str(e)
