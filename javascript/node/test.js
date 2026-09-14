const { hello, InferenceSession } = require('./index.js');
console.assert(hello() === 'pmml-runtime');
const fs = require('fs');
const candidates = ['../../bench/pmml/DecisionTreeIris.pmml', 'bench/pmml/DecisionTreeIris.pmml', '../bench/pmml/DecisionTreeIris.pmml'];
const path = candidates.find(p => fs.existsSync(p));
const s = new InferenceSession(path);
const out = s.run({ 'Petal.Length': 1.4 });
console.assert(out && ('predictedValue' in out));
console.log('ok');
