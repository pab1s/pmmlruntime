//! _native — pyo3 InferenceSession over pmmlruntime::Session (ORT python pattern)
use std::collections::HashMap;
use std::path::Path;

use pmmlruntime::base::{SymbolId, Value};
use pmmlruntime::session::batch::Batch;
use pmmlruntime::session::{PmmlEnv, Session};
use pmmlruntime::session::SessionOptions as RustSessionOptions;
use pmmlruntime::session::GraphOptimizationLevel as RustGraphLevel;
use pyo3::prelude::*;
use pyo3::types::{PyBool, PyBytes, PyDict, PyList, PyString};

#[pyfunction]
fn hello() -> PyResult<String> {
    Ok("pmml-runtime".into())
}

fn pyany_to_value(obj: &Bound<'_, PyAny>, sess: &Session, field_name: &str) -> PyResult<Value> {
    if obj.is_none() {
        return Ok(Value::Missing);
    }
    if obj.is_instance_of::<PyBool>() {
        let b: bool = obj.extract()?;
        return Ok(Value::Continuous(if b { 1.0 } else { 0.0 }));
    }
    // String — categorical/discrete or numeric string via sess.string_to_value
    if obj.is_instance_of::<PyString>() {
        let s: String = obj.extract()?;
        if s.is_empty() {
            return Ok(Value::Missing);
        }
        return Ok(sess.string_to_value(field_name, &s));
    }
    // Try float/int (including numpy scalars that can extract as f64)
    if let Ok(f) = obj.extract::<f64>() {
        return Ok(Value::Continuous(f));
    }
    // Fallback try __float__
    if let Ok(f) = obj.call_method0("__float__").and_then(|v| v.extract::<f64>()) {
        return Ok(Value::Continuous(f));
    }
    Ok(Value::Missing)
}

fn value_to_pyobject(py: Python, v: Value, sess: &Session) -> PyObject {
    use pyo3::IntoPy;
    match v {
        Value::Missing => py.None(),
        Value::Continuous(f) => f.into_py(py),
        Value::Discrete(SymbolId(id)) => {
            let s = sess
                .ir
                .symbol_names
                .get(&SymbolId(id))
                .cloned()
                .unwrap_or_else(|| format!("Symbol({})", id));
            s.into_py(py)
        }
    }
}

#[pyclass]
struct InferenceSession {
    session: Session,
    _env: PmmlEnv,
}

#[pymethods]
impl InferenceSession {
    #[new]
    #[pyo3(signature = (path_or_bytes, sess_options=None, providers=None, provider_options=None))]
    fn new(
        path_or_bytes: Bound<'_, PyAny>,
        sess_options: Option<Bound<'_, PyAny>>,
        providers: Option<Bound<'_, PyAny>>,
        provider_options: Option<Bound<'_, PyAny>>,
    ) -> PyResult<Self> {
        let _ = sess_options;
        let _ = providers;
        let _ = provider_options;
        let env = PmmlEnv::new();
        let opts = RustSessionOptions::default();
        let session = if let Ok(s) = path_or_bytes.extract::<String>() {
            let p = Path::new(&s);
            if p.exists() {
                Session::from_file(&env, &s, opts).map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?
            } else if s.len() > 200 && s.contains("<PMML") {
                Session::from_bytes(&env, s.as_bytes(), opts).map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?
            } else {
                return Err(PyErr::new::<pyo3::exceptions::PyFileNotFoundError, _>(format!("model not found: {}", s)));
            }
        } else if let Ok(b) = path_or_bytes.downcast::<PyBytes>() {
            let bytes = b.as_bytes();
            Session::from_bytes(&env, bytes, opts).map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?
        } else if let Ok(b) = path_or_bytes.extract::<Vec<u8>>() {
            Session::from_bytes(&env, &b, opts).map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?
        } else if let Ok(path_str) = path_or_bytes.call_method0("__fspath__").and_then(|v| v.extract::<String>()) {
            if Path::new(&path_str).exists() {
                Session::from_file(&env, &path_str, opts).map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?
            } else {
                return Err(PyErr::new::<pyo3::exceptions::PyFileNotFoundError, _>(format!("model not found via fspath: {}", path_str)));
            }
        } else {
            return Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>("path_or_bytes must be str, bytes or Path"));
        };
        Ok(Self { session, _env: env })
    }

    #[staticmethod]
    #[pyo3(signature = (bytes, sess_options=None))]
    fn from_bytes(bytes: Vec<u8>, sess_options: Option<Bound<'_, PyAny>>) -> PyResult<Self> {
        let _ = sess_options;
        let env = PmmlEnv::new();
        let opts = RustSessionOptions::default();
        let sess = Session::from_bytes(&env, &bytes, opts).map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
        Ok(Self { session: sess, _env: env })
    }

    fn get_inputs(&self) -> PyResult<Vec<HashMap<String, String>>> {
        let mut out = Vec::new();
        for (fid, name) in &self.session.ir.field_names {
            let meta = self.session.ir.data_dictionary.iter().find(|m| &m.field_id == fid);
            let dtype = meta.map(|m| format!("{:?}", m.data_type)).unwrap_or_else(|| "unknown".into());
            let optype = meta.map(|m| format!("{:?}", m.op_type)).unwrap_or_else(|| "unknown".into());
            let mut m = HashMap::new();
            m.insert("name".to_string(), name.clone());
            m.insert("type".to_string(), dtype);
            m.insert("opType".to_string(), optype);
            out.push(m);
        }
        Ok(out)
    }

    fn get_outputs(&self) -> PyResult<Vec<HashMap<String, String>>> {
        let mut out = Vec::new();
        let outs: Vec<pmmlruntime::ir::OutputFieldIr> = match &self.session.ir.model {
            pmmlruntime::ir::ModelIr::Tree(t) => t.output.clone(),
            pmmlruntime::ir::ModelIr::Regression(r) => r.output.clone(),
            pmmlruntime::ir::ModelIr::Mining(m) => m.output.clone(),
            pmmlruntime::ir::ModelIr::Scorecard(s) => s.output.clone(),
            pmmlruntime::ir::ModelIr::Clustering(c) => c.output.clone(),
            pmmlruntime::ir::ModelIr::NaiveBayes(n) => n.output.clone(),
            pmmlruntime::ir::ModelIr::NearestNeighbor(n) => n.output.clone(),
            pmmlruntime::ir::ModelIr::SupportVectorMachine(s) => s.output.clone(),
            pmmlruntime::ir::ModelIr::GeneralRegression(g) => g.output.clone(),
            pmmlruntime::ir::ModelIr::Association(a) => a.output.clone(),
            pmmlruntime::ir::ModelIr::RuleSet(r) => r.output.clone(),
            pmmlruntime::ir::ModelIr::NeuralNetwork(n) => n.output.clone(),
            pmmlruntime::ir::ModelIr::AnomalyDetection(a) => a.output.clone(),
            pmmlruntime::ir::ModelIr::Baseline(b) => b.output.clone(),
            pmmlruntime::ir::ModelIr::GaussianProcess(g) => g.output.clone(),
            pmmlruntime::ir::ModelIr::Text(t) => t.output.clone(),
            pmmlruntime::ir::ModelIr::TimeSeries(t) => t.output.clone(),
            pmmlruntime::ir::ModelIr::Sequence(s) => s.output.clone(),
            pmmlruntime::ir::ModelIr::BayesianNetwork(b) => b.output.clone(),
        };
        if outs.is_empty() {
            let mut m = HashMap::new();
            m.insert("name".to_string(), "predictedValue".to_string());
            out.push(m);
        } else {
            for o in outs {
                let mut m = HashMap::new();
                m.insert("name".to_string(), o.name.clone());
                m.insert("feature".to_string(), format!("{:?}", o.feature));
                out.push(m);
            }
        }
        Ok(out)
    }

    fn get_modelmeta(&self) -> PyResult<HashMap<String, String>> {
        let mut m = HashMap::new();
        m.insert("pmml_version".to_string(), "4.4".to_string());
        let model_type = match &self.session.ir.model {
            pmmlruntime::ir::ModelIr::Tree(_) => "TreeModel",
            pmmlruntime::ir::ModelIr::Regression(_) => "RegressionModel",
            pmmlruntime::ir::ModelIr::Mining(_) => "MiningModel",
            pmmlruntime::ir::ModelIr::Scorecard(_) => "Scorecard",
            pmmlruntime::ir::ModelIr::Clustering(_) => "ClusteringModel",
            pmmlruntime::ir::ModelIr::NaiveBayes(_) => "NaiveBayesModel",
            pmmlruntime::ir::ModelIr::NearestNeighbor(_) => "NearestNeighborModel",
            pmmlruntime::ir::ModelIr::SupportVectorMachine(_) => "SupportVectorMachineModel",
            pmmlruntime::ir::ModelIr::GeneralRegression(_) => "GeneralRegressionModel",
            pmmlruntime::ir::ModelIr::Association(_) => "AssociationModel",
            pmmlruntime::ir::ModelIr::RuleSet(_) => "RuleSetModel",
            pmmlruntime::ir::ModelIr::NeuralNetwork(_) => "NeuralNetwork",
            pmmlruntime::ir::ModelIr::AnomalyDetection(_) => "AnomalyDetectionModel",
            pmmlruntime::ir::ModelIr::Baseline(_) => "BaselineModel",
            pmmlruntime::ir::ModelIr::GaussianProcess(_) => "GaussianProcessModel",
            pmmlruntime::ir::ModelIr::Text(_) => "TextModel",
            pmmlruntime::ir::ModelIr::TimeSeries(_) => "TimeSeriesModel",
            pmmlruntime::ir::ModelIr::Sequence(_) => "SequenceModel",
            pmmlruntime::ir::ModelIr::BayesianNetwork(_) => "BayesianNetworkModel",
        };
        m.insert("model_type".to_string(), model_type.to_string());
        m.insert("producer".to_string(), "pmmlruntime".to_string());
        Ok(m)
    }

    #[pyo3(signature = (output_names, input_feed, run_options=None))]
    fn run(
        &self,
        py: Python<'_>,
        output_names: Option<Vec<String>>,
        input_feed: Bound<'_, PyAny>,
        run_options: Option<Bound<'_, PyAny>>,
    ) -> PyResult<PyObject> {
        let _ = run_options;
        let _ = output_names;
        if let Ok(dict) = input_feed.downcast::<PyDict>() {
            let mut map = HashMap::new();
            for (k, v) in dict.iter() {
                let key = k.extract::<String>()?;
                let val = pyany_to_value(&v, &self.session, &key)?;
                map.insert(key, val);
            }
            let batch: &dyn Batch = &map as &dyn Batch;
            let result = py.allow_threads(|| self.session.run(batch))
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
            let rows = result.into_rows();
            let pylist = PyList::empty_bound(py);
            for row in rows {
                let d = PyDict::new_bound(py);
                for (k, v) in row {
                    d.set_item(k, value_to_pyobject(py, v, &self.session))?;
                }
                pylist.append(d)?;
            }
            return Ok(pylist.unbind().into());
        }
        if let Ok(list) = input_feed.downcast::<PyList>() {
            let mut batch: Vec<HashMap<String, Value>> = Vec::with_capacity(list.len());
            for item in list.iter() {
                let d = item.downcast::<PyDict>().map_err(|_| PyErr::new::<pyo3::exceptions::PyTypeError, _>("batch items must be dict"))?;
                let mut map = HashMap::new();
                for (k, v) in d.iter() {
                    let key = k.extract::<String>()?;
                    let val = pyany_to_value(&v, &self.session, &key)?;
                    map.insert(key, val);
                }
                batch.push(map);
            }
            let result = py.allow_threads(|| self.session.run(&batch as &dyn Batch))
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;
            let rows = result.into_rows();
            let pylist = PyList::empty_bound(py);
            for row in rows {
                let d = PyDict::new_bound(py);
                for (k, v) in row {
                    d.set_item(k, value_to_pyobject(py, v, &self.session))?;
                }
                pylist.append(d)?;
            }
            return Ok(pylist.unbind().into());
        }
        Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>("input_feed must be dict or list[dict] (pyarrow Table via RunArrow TODO)"))
    }

    fn run_with_iobinding(&self, _py: Python<'_>, _binding: Bound<'_, PyAny>) -> PyResult<PyObject> {
        Err(PyErr::new::<pyo3::exceptions::PyNotImplementedError, _>("io_binding not yet implemented — use run()"))
    }

    fn io_binding(&self, _py: Python<'_>) -> PyResult<PyObject> {
        Err(PyErr::new::<pyo3::exceptions::PyNotImplementedError, _>("io_binding not yet implemented"))
    }
}

#[pyclass]
struct PySessionOptions {
    inner: RustSessionOptions,
}

#[pymethods]
impl PySessionOptions {
    #[new]
    fn new() -> Self { Self { inner: RustSessionOptions::default() } }

    #[getter]
    fn get_graph_optimization_level(&self) -> i32 { self.inner.graph_optimization_level as i32 }
    #[setter]
    fn set_graph_optimization_level(&mut self, lvl: i32) {
        let l = match lvl {
            0 => RustGraphLevel::DisableAll,
            1 => RustGraphLevel::EnableBasic,
            2 => RustGraphLevel::EnableExtended,
            3 => RustGraphLevel::EnableAll,
            _ => RustGraphLevel::EnableBasic,
        };
        self.inner = self.inner.clone().graph_optimization_level(l);
    }
}

#[pyclass]
struct GraphOptimizationLevelCls;

#[pymethods]
impl GraphOptimizationLevelCls {
    #[classattr] const ORT_DISABLE_ALL: i32 = 0;
    #[classattr] const ORT_ENABLE_BASIC: i32 = 1;
    #[classattr] const ORT_ENABLE_EXTENDED: i32 = 2;
    #[classattr] const ORT_ENABLE_ALL: i32 = 3;
}

#[pymodule]
fn _native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(hello, m)?)?;
    m.add_class::<InferenceSession>()?;
    m.add_class::<PySessionOptions>()?;
    m.add_class::<GraphOptimizationLevelCls>()?;
    Ok(())
}
