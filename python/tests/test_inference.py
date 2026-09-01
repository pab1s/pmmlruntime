import pmmlruntime as pm

def test_hello():
    assert pm.hello() == "pmml-runtime"

# TODO: after PmmlApi linking
# def test_inference_session():
#     sess = pm.InferenceSession("bench/pmml/DecisionTreeIris.pmml")
#     assert len(sess.get_inputs()) == 4
#     out = sess.run(None, {"Petal.Length": 1.4, "Petal.Width": 0.2, "Sepal.Length": 5.1, "Sepal.Width": 3.5})
#     assert "predictedValue" in out[0]
