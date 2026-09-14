import pmmlruntime as pm
from pathlib import Path

def test_hello():
    assert pm.hello() == "pmml-runtime"


def test_inference_session_iris():
    root = Path(__file__).resolve().parents[2]
    sess = pm.InferenceSession(str(root / "bench/pmml/DecisionTreeIris.pmml"))
    assert any(f["name"] == "Petal.Length" for f in sess.get_inputs())
    out = sess.run(None, {"Petal.Length": 1.4, "Petal.Width": 0.2})
    assert "predictedValue" in out[0]
    out2 = sess.run(["predictedValue"], {"Petal.Length": 1.4, "Petal.Width": 0.2})
    assert list(out2[0].keys()) == ["predictedValue"]
