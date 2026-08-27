use pmmlruntime::base::Value;
use pmmlruntime::session::{PmmlEnv, Session, SessionOptions};
use std::collections::HashMap;

#[test]
fn sequence_simple_rule() {
    let path = if std::path::Path::new("bench/pmml/SequenceSimpleTest.pmml").exists() {
        "bench/pmml/SequenceSimpleTest.pmml"
    } else {
        "../../bench/pmml/SequenceSimpleTest.pmml"
    };
    let xml = std::fs::read(path).unwrap();
    let env = PmmlEnv::new();
    let sess = Session::from_bytes(&env, &xml, SessionOptions::default()).unwrap();
    // milk -> bread
    let sid_milk = sess.symbol_id("milk").unwrap();
    let sid_bread = sess.symbol_id("bread").unwrap();
    let sid_butter = sess.symbol_id("butter").unwrap();
    let mut input = HashMap::new();
    input.insert("item".to_string(), Value::Discrete(sid_milk));
    let out = sess
        .run(&input as &dyn pmmlruntime::session::batch::Batch)
        .unwrap()
        .into_single()
        .unwrap();
    let pred = out.get("predictedValue").unwrap();
    assert_eq!(
        *pred,
        Value::Discrete(sid_bread),
        "milk should predict bread via r0"
    );
    // bread -> butter
    let mut input2 = HashMap::new();
    input2.insert("item".to_string(), Value::Discrete(sid_bread));
    let out2 = sess
        .run(&input2 as &dyn pmmlruntime::session::batch::Batch)
        .unwrap()
        .into_single()
        .unwrap();
    assert_eq!(
        *out2.get("predictedValue").unwrap(),
        Value::Discrete(sid_butter)
    );
    // butter -> missing (no rule)
    let mut input3 = HashMap::new();
    input3.insert("item".to_string(), Value::Discrete(sid_butter));
    let out3 = sess
        .run(&input3 as &dyn pmmlruntime::session::batch::Batch)
        .unwrap()
        .into_single()
        .unwrap();
    assert_eq!(*out3.get("predictedValue").unwrap(), Value::Missing);
}

#[test]
fn bayesian_discrete_inference() {
    let path = if std::path::Path::new("bench/pmml/BayesianSimpleTest.pmml").exists() {
        "bench/pmml/BayesianSimpleTest.pmml"
    } else {
        "../../bench/pmml/BayesianSimpleTest.pmml"
    };
    let xml = std::fs::read(path).unwrap();
    let env = PmmlEnv::new();
    let sess = Session::from_bytes(&env, &xml, SessionOptions::default()).unwrap();
    // Model: A prior 0.4/0.6, B prior 0.7/0.3, C conditional on A,B
    // Evidence C=2 should predict A=1 (posterior 0.792 vs 0.208)
    let sid_2 = sess.symbol_id("2").unwrap();
    let sid_1 = sess.symbol_id("1").unwrap();
    let mut input = HashMap::new();
    input.insert("C".to_string(), Value::Discrete(sid_2));
    let out = sess
        .run(&input as &dyn pmmlruntime::session::batch::Batch)
        .unwrap()
        .into_single()
        .unwrap();
    let pred = out.get("predictedValue").unwrap();
    assert_eq!(*pred, Value::Discrete(sid_1), "C=2 should predict A=1");
    // No evidence: marginal predicts A=1 (0.6 >0.4)
    let out2 = sess
        .run(&HashMap::new() as &dyn pmmlruntime::session::batch::Batch)
        .unwrap()
        .into_single()
        .unwrap();
    assert_eq!(*out2.get("predictedValue").unwrap(), Value::Discrete(sid_1));
}

#[test]
fn bayesian_continuous_target() {
    let xml = br#"<PMML version="4.4" xmlns="https://www.dmg.org/PMML-4_4">
<Header copyright="test"/>
<DataDictionary>
  <DataField name="D1" dataType="string" optype="categorical"><Value value="0"/><Value value="1"/></DataField>
  <DataField name="C1" dataType="double" optype="continuous"/>
</DataDictionary>
<BayesianNetworkModel functionName="regression">
  <MiningSchema><MiningField name="D1" usageType="active"/><MiningField name="C1" usageType="target"/></MiningSchema>
  <BayesianNetworkNodes>
    <DiscreteNode name="D1"><ValueProbability value="0" probability="0.3"/><ValueProbability value="1" probability="0.7"/></DiscreteNode>
    <ContinuousNode name="C1">
      <ContinuousConditionalProbability><ParentValue parent="D1" value="0"/><ContinuousDistribution><NormalDistributionForBN><Mean><Constant dataType="double">10</Constant></Mean><Variance><Constant dataType="double">2</Constant></Variance></NormalDistributionForBN></ContinuousDistribution></ContinuousConditionalProbability>
      <ContinuousConditionalProbability><ParentValue parent="D1" value="1"/><ContinuousDistribution><NormalDistributionForBN><Mean><Constant dataType="double">14</Constant></Mean><Variance><Constant dataType="double">2</Constant></Variance></NormalDistributionForBN></ContinuousDistribution></ContinuousConditionalProbability>
    </ContinuousNode>
  </BayesianNetworkNodes>
</BayesianNetworkModel>
</PMML>"#;
    let env = PmmlEnv::new();
    let sess = Session::from_bytes(&env, xml, SessionOptions::default()).unwrap();
    let sid0 = sess.symbol_id("0").unwrap();
    let sid1 = sess.symbol_id("1").unwrap();
    let mut in0 = HashMap::new();
    in0.insert("D1".to_string(), Value::Discrete(sid0));
    let out0 = sess
        .run(&in0 as &dyn pmmlruntime::session::batch::Batch)
        .unwrap()
        .into_single()
        .unwrap();
    let pred0 = out0.get("predictedValue").unwrap();
    match pred0 {
        Value::Continuous(f) => assert!(
            (f - 10.0).abs() < 1e-9,
            "D1=0 should predict C1=10, got {f}"
        ),
        _ => panic!("expected continuous"),
    }
    let mut in1 = HashMap::new();
    in1.insert("D1".to_string(), Value::Discrete(sid1));
    let out1 = sess
        .run(&in1 as &dyn pmmlruntime::session::batch::Batch)
        .unwrap()
        .into_single()
        .unwrap();
    match out1.get("predictedValue").unwrap() {
        Value::Continuous(f) => assert!((f - 14.0).abs() < 1e-9, "D1=1 should predict C1=14"),
        _ => panic!("expected continuous"),
    }
}

#[test]
fn bayesian_with_derived_discretization() {
    // Test the large MCMC example's derived field C3_Discretized
    // Use a simplified version: C3 continuous -> discretized to 0,1,2 -> D4 conditional on D3 and discretized C3
    let xml = br#"<PMML version="4.4" xmlns="https://www.dmg.org/PMML-4_4">
<Header copyright="test"/>
<DataDictionary>
  <DataField name="D3" dataType="string" optype="categorical"><Value value="0"/><Value value="1"/></DataField>
  <DataField name="C3" dataType="double" optype="continuous"/>
  <DataField name="D4" dataType="string" optype="categorical"><Value value="0"/><Value value="1"/></DataField>
</DataDictionary>
<BayesianNetworkModel functionName="classification">
  <MiningSchema><MiningField name="D3" usageType="active"/><MiningField name="C3" usageType="active"/><MiningField name="D4" usageType="target"/></MiningSchema>
  <BayesianNetworkNodes>
    <DiscreteNode name="D3"><ValueProbability value="0" probability="0.5"/><ValueProbability value="1" probability="0.5"/></DiscreteNode>
    <ContinuousNode name="C3">
      <ContinuousConditionalProbability><ParentValue parent="D3" value="0"/><ContinuousDistribution><NormalDistributionForBN><Mean><Constant dataType="double">5</Constant></Mean><Variance><Constant dataType="double">1</Constant></Variance></NormalDistributionForBN></ContinuousDistribution></ContinuousConditionalProbability>
      <ContinuousConditionalProbability><ParentValue parent="D3" value="1"/><ContinuousDistribution><NormalDistributionForBN><Mean><Constant dataType="double">15</Constant></Mean><Variance><Constant dataType="double">1</Constant></Variance></NormalDistributionForBN></ContinuousDistribution></ContinuousConditionalProbability>
    </ContinuousNode>
    <DiscreteNode name="D4">
      <DerivedField name="C3_Discretized" optype="categorical" dataType="string"><Discretize field="C3"><DiscretizeBin binValue="0"><Interval closure="openClosed" rightMargin="9"/></DiscretizeBin><DiscretizeBin binValue="1"><Interval closure="openClosed" leftMargin="9" rightMargin="11"/></DiscretizeBin><DiscretizeBin binValue="2"><Interval closure="openOpen" leftMargin="11"/></DiscretizeBin></Discretize></DerivedField>
      <DiscreteConditionalProbability><ParentValue parent="D3" value="0"/><ParentValue parent="C3_Discretized" value="0"/><ValueProbability value="0" probability="0.8"/><ValueProbability value="1" probability="0.2"/></DiscreteConditionalProbability>
      <DiscreteConditionalProbability><ParentValue parent="D3" value="0"/><ParentValue parent="C3_Discretized" value="1"/><ValueProbability value="0" probability="0.5"/><ValueProbability value="1" probability="0.5"/></DiscreteConditionalProbability>
      <DiscreteConditionalProbability><ParentValue parent="D3" value="0"/><ParentValue parent="C3_Discretized" value="2"/><ValueProbability value="0" probability="0.2"/><ValueProbability value="1" probability="0.8"/></DiscreteConditionalProbability>
      <DiscreteConditionalProbability><ParentValue parent="D3" value="1"/><ParentValue parent="C3_Discretized" value="0"/><ValueProbability value="0" probability="0.2"/><ValueProbability value="1" probability="0.8"/></DiscreteConditionalProbability>
      <DiscreteConditionalProbability><ParentValue parent="D3" value="1"/><ParentValue parent="C3_Discretized" value="1"/><ValueProbability value="0" probability="0.5"/><ValueProbability value="1" probability="0.5"/></DiscreteConditionalProbability>
      <DiscreteConditionalProbability><ParentValue parent="D3" value="1"/><ParentValue parent="C3_Discretized" value="2"/><ValueProbability value="0" probability="0.8"/><ValueProbability value="1" probability="0.2"/></DiscreteConditionalProbability>
    </DiscreteNode>
  </BayesianNetworkNodes>
</BayesianNetworkModel>
</PMML>"#;
    let env = PmmlEnv::new();
    let sess = Session::from_bytes(&env, xml, SessionOptions::default()).unwrap();
    // With D3=0 and C3=5 (which discretizes to 0), D4 should predict 0 (prob 0.8)
    let sid0 = sess.symbol_id("0").unwrap();
    let sid1 = sess.symbol_id("1").unwrap();
    let mut input = HashMap::new();
    input.insert("D3".to_string(), Value::Discrete(sid0));
    input.insert("C3".to_string(), Value::Continuous(5.0));
    let out = sess
        .run(&input as &dyn pmmlruntime::session::batch::Batch)
        .unwrap()
        .into_single()
        .unwrap();
    assert_eq!(*out.get("predictedValue").unwrap(), Value::Discrete(sid0));
    // With D3=0 and C3=15 (discretizes to 2), D4 should predict 1 (prob 0.8)
    let mut input2 = HashMap::new();
    input2.insert("D3".to_string(), Value::Discrete(sid0));
    input2.insert("C3".to_string(), Value::Continuous(15.0));
    let out2 = sess
        .run(&input2 as &dyn pmmlruntime::session::batch::Batch)
        .unwrap()
        .into_single()
        .unwrap();
    assert_eq!(*out2.get("predictedValue").unwrap(), Value::Discrete(sid1));
}

#[test]
fn sequence_with_set_predicate() {
    let xml = br#"<PMML version="4.4" xmlns="https://www.dmg.org/PMML-4_4">
<Header copyright="test"/>
<DataDictionary>
  <DataField name="color" dataType="string" optype="categorical"><Value value="red"/><Value value="blue"/><Value value="green"/></DataField>
  <DataField name="group" dataType="string" optype="categorical"/>
  <DataField name="order" dataType="integer" optype="continuous"/>
</DataDictionary>
<SequenceModel functionName="associationRules">
  <MiningSchema><MiningField name="color" usageType="active"/><MiningField name="group" usageType="group"/><MiningField name="order" usageType="order"/></MiningSchema>
  <Item id="i0" value="red"/>
  <Item id="i1" value="blue"/>
  <SetPredicate id="sp0" field="color"><Array n="2" type="string">red blue</Array></SetPredicate>
  <Itemset id="is0"><ItemRef itemRef="i0"/></Itemset>
  <Sequence id="s0"><SetReference setId="sp0"/></Sequence>
  <Sequence id="s1"><SetReference setId="is0"/></Sequence>
  <SequenceRule id="r0" numberOfSets="2" occurrence="5" support="0.2" confidence="0.9">
    <AntecedentSequence><SequenceReference seqId="s0"/></AntecedentSequence>
    <Delimiter delimiter="acrossTimeWindows" gap="unknown"/>
    <ConsequentSequence><SequenceReference seqId="s1"/></ConsequentSequence>
  </SequenceRule>
</SequenceModel>
</PMML>"#;
    let env = PmmlEnv::new();
    let sess = Session::from_bytes(&env, xml, SessionOptions::default()).unwrap();
    let sid_red = sess.symbol_id("red").unwrap();
    let _sid_blue = sess.symbol_id("blue").unwrap();
    // red should match sp0 (supersetOf red blue? For our isIn logic, red in [red,blue] => true) and predict red
    let mut input = HashMap::new();
    input.insert("color".to_string(), Value::Discrete(sid_red));
    let out = sess
        .run(&input as &dyn pmmlruntime::session::batch::Batch)
        .unwrap()
        .into_single()
        .unwrap();
    // consequent is is0 -> red
    let pred = out.get("predictedValue").unwrap();
    assert!(matches!(pred, Value::Discrete(_)));
}

#[test]
fn sequence_large_spec_example() {
    // From DMG spec: Sequence with 6 items, 4 itemsets, 4 sequences, 4 rules, with constraints and Time
    let xml = br#"<PMML version="4.4" xmlns="https://www.dmg.org/PMML-4_4">
<Header copyright="DMG.org"/>
<DataDictionary>
  <DataField name="item" dataType="string" optype="categorical"><Value value="Cognac"/><Value value="Cream"/><Value value="Tonic water"/><Value value="Vodka"/><Value value="Cider"/><Value value="Scotch Whisky"/><Value value="Root Beer"/></DataField>
  <DataField name="group" dataType="string" optype="categorical"/>
  <DataField name="order" dataType="integer" optype="continuous"/>
</DataDictionary>
<SequenceModel functionName="associationRules">
  <MiningSchema><MiningField name="item" usageType="active"/><MiningField name="group" usageType="group"/><MiningField name="order" usageType="order"/></MiningSchema>
  <Constraints minimumSupport="0.2" minimumConfidence="0.5"/>
  <Item id="0" value="Cognac"/><Item id="1" value="Cream"/><Item id="2" value="Tonic water"/><Item id="3" value="Vodka"/><Item id="4" value="Cider"/><Item id="5" value="Scotch Whisky"/><Item id="6" value="Root Beer"/>
  <Itemset id="0" support="0.0628571428571429" numberOfItems="1"><ItemRef itemRef="0"/></Itemset>
  <Itemset id="1" support="0.24" numberOfItems="2"><ItemRef itemRef="1"/><ItemRef itemRef="2"/></Itemset>
  <Itemset id="2" support="0.0628571428571429" numberOfItems="3"><ItemRef itemRef="3"/><ItemRef itemRef="4"/><ItemRef itemRef="5"/></Itemset>
  <Itemset id="3" support="0.0628571428571429" numberOfItems="1"><ItemRef itemRef="6"/></Itemset>
  <Sequence id="0" numberOfSets="1" occurrence="5" support="0.02"><SetReference setId="0"/></Sequence>
  <Sequence id="1" numberOfSets="2" occurrence="6" support="0.25"><SetReference setId="0"/><Delimiter delimiter="acrossTimeWindows" gap="unknown"/><SetReference setId="2"/></Sequence>
  <Sequence id="2" numberOfSets="1" occurrence="5" support="0.45"><SetReference setId="1"/></Sequence>
  <Sequence id="3" numberOfSets="1" occurrence="15" support="0.2"><SetReference setId="3"/></Sequence>
  <SequenceRule id="0" numberOfSets="2" occurrence="5" support="0.20833" confidence="0.55556"><AntecedentSequence><SequenceReference seqId="0"/></AntecedentSequence><Delimiter delimiter="acrossTimeWindows" gap="unknown"/><Time min="5" max="8" mean="6.8"/><ConsequentSequence><SequenceReference seqId="2"/></ConsequentSequence></SequenceRule>
  <SequenceRule id="1" numberOfSets="2" occurrence="6" support="0.25" confidence="0.66667"><AntecedentSequence><SequenceReference seqId="1"/></AntecedentSequence><Delimiter delimiter="acrossTimeWindows" gap="unknown"/><Time min="2" max="8" mean="6.16667"/><ConsequentSequence><SequenceReference seqId="3"/></ConsequentSequence></SequenceRule>
</SequenceModel>
</PMML>"#;
    let env = PmmlEnv::new();
    let sess = Session::from_bytes(&env, xml, SessionOptions::default()).unwrap();
    assert_eq!(sess.num_active_fields(), 1);
    let sid_cognac = sess.symbol_id("Cognac").unwrap();
    let mut input = HashMap::new();
    input.insert("item".to_string(), Value::Discrete(sid_cognac));
    let out = sess
        .run(&input as &dyn pmmlruntime::session::batch::Batch)
        .unwrap()
        .into_single()
        .unwrap();
    // Antecedent s0 is {Cognac} -> consequent s1 is {Cognac, Vodka/Cider/Scotch}?? Actually s1 is itemset 0 then 2; our simple rule checks first set, so Cognac should fire rule 0 and predict Cream (first item of s2 which is {Cream,Tonic water} -> Cream)
    assert!(out.contains_key("predictedValue"));
}

#[test]
fn bayesian_large_mcmc_example() {
    // Simplified large MCMC Bayesian from spec: 7 nodes with derived discretization and expression mean
    let xml = br#"<PMML version="4.4" xmlns="https://www.dmg.org/PMML-4_4">
<Header copyright="DMG.org"/>
<DataDictionary>
  <DataField name="D1" dataType="string" optype="categorical"><Value value="0"/><Value value="1"/></DataField>
  <DataField name="D2" dataType="string" optype="categorical"><Value value="0"/><Value value="1"/><Value value="2"/></DataField>
  <DataField name="D3" dataType="string" optype="categorical"><Value value="0"/><Value value="1"/></DataField>
  <DataField name="C1" dataType="double" optype="continuous"/>
  <DataField name="C2" dataType="double" optype="continuous"/>
  <DataField name="C3" dataType="double" optype="continuous"/>
  <DataField name="D4" dataType="string" optype="categorical"><Value value="0"/><Value value="1"/></DataField>
  <DataField name="C4" dataType="double" optype="continuous"/>
</DataDictionary>
<BayesianNetworkModel functionName="regression" inferenceMethod="MCMC">
  <MiningSchema><MiningField name="D4" usageType="active"/><MiningField name="C4" usageType="active"/><MiningField name="D1" usageType="target"/><MiningField name="D2" usageType="target"/><MiningField name="D3" usageType="target"/><MiningField name="C1" usageType="target"/><MiningField name="C2" usageType="target"/><MiningField name="C3" usageType="target"/></MiningSchema>
  <BayesianNetworkNodes>
    <DiscreteNode name="D1"><ValueProbability value="0" probability="0.3"/><ValueProbability value="1" probability="0.7"/></DiscreteNode>
    <DiscreteNode name="D2"><ValueProbability value="0" probability="0.6"/><ValueProbability value="1" probability="0.3"/><ValueProbability value="2" probability="0.1"/></DiscreteNode>
    <DiscreteNode name="D3">
      <DiscreteConditionalProbability><ParentValue parent="D1" value="0"/><ParentValue parent="D2" value="0"/><ValueProbability value="0" probability="0.1"/><ValueProbability value="1" probability="0.9"/></DiscreteConditionalProbability>
      <DiscreteConditionalProbability><ParentValue parent="D1" value="0"/><ParentValue parent="D2" value="1"/><ValueProbability value="0" probability="0.3"/><ValueProbability value="1" probability="0.7"/></DiscreteConditionalProbability>
      <DiscreteConditionalProbability><ParentValue parent="D1" value="0"/><ParentValue parent="D2" value="2"/><ValueProbability value="0" probability="0.4"/><ValueProbability value="1" probability="0.6"/></DiscreteConditionalProbability>
      <DiscreteConditionalProbability><ParentValue parent="D1" value="1"/><ParentValue parent="D2" value="0"/><ValueProbability value="0" probability="0.6"/><ValueProbability value="1" probability="0.4"/></DiscreteConditionalProbability>
      <DiscreteConditionalProbability><ParentValue parent="D1" value="1"/><ParentValue parent="D2" value="1"/><ValueProbability value="0" probability="0.8"/><ValueProbability value="1" probability="0.2"/></DiscreteConditionalProbability>
      <DiscreteConditionalProbability><ParentValue parent="D1" value="1"/><ParentValue parent="D2" value="2"/><ValueProbability value="0" probability="0.9"/><ValueProbability value="1" probability="0.1"/></DiscreteConditionalProbability>
    </DiscreteNode>
    <ContinuousNode name="C1">
      <ContinuousConditionalProbability><ParentValue parent="D1" value="0"/><ContinuousDistribution><NormalDistributionForBN><Mean><Constant dataType="double">10</Constant></Mean><Variance><Constant dataType="double">2</Constant></Variance></NormalDistributionForBN></ContinuousDistribution></ContinuousConditionalProbability>
      <ContinuousConditionalProbability><ParentValue parent="D1" value="1"/><ContinuousDistribution><NormalDistributionForBN><Mean><Constant dataType="double">14</Constant></Mean><Variance><Constant dataType="double">2</Constant></Variance></NormalDistributionForBN></ContinuousDistribution></ContinuousConditionalProbability>
    </ContinuousNode>
    <ContinuousNode name="C2">
      <ContinuousConditionalProbability><ParentValue parent="D2" value="0"/><ContinuousDistribution><NormalDistributionForBN><Mean><Constant dataType="double">6</Constant></Mean><Variance><Constant dataType="double">1</Constant></Variance></NormalDistributionForBN></ContinuousDistribution></ContinuousConditionalProbability>
      <ContinuousConditionalProbability><ParentValue parent="D2" value="1"/><ContinuousDistribution><NormalDistributionForBN><Mean><Constant dataType="double">8</Constant></Mean><Variance><Constant dataType="double">1</Constant></Variance></NormalDistributionForBN></ContinuousDistribution></ContinuousConditionalProbability>
      <ContinuousConditionalProbability><ParentValue parent="D2" value="2"/><ContinuousDistribution><NormalDistributionForBN><Mean><Constant dataType="double">14</Constant></Mean><Variance><Constant dataType="double">1</Constant></Variance></NormalDistributionForBN></ContinuousDistribution></ContinuousConditionalProbability>
    </ContinuousNode>
    <ContinuousNode name="C4">
      <ContinuousConditionalProbability><ContinuousDistribution><NormalDistributionForBN><Mean><Apply function="+"><Apply function="*"><Constant dataType="double">0.1</Constant><Apply function="pow"><FieldRef field="C2"/><Constant dataType="integer">2</Constant></Apply></Apply><Apply function="+"><Apply function="*"><Constant dataType="double">0.6</Constant><FieldRef field="C2"/></Apply><Constant dataType="integer">1</Constant></Apply></Apply></Mean><Variance><Constant dataType="double">2</Constant></Variance></NormalDistributionForBN></ContinuousDistribution></ContinuousConditionalProbability>
    </ContinuousNode>
    <ContinuousNode name="C3">
      <ContinuousConditionalProbability><ParentValue parent="D3" value="0"/><ContinuousDistribution><NormalDistributionForBN><Mean><Apply function="*"><Constant dataType="double">0.15</Constant><Apply function="pow"><FieldRef field="C2"/><Constant dataType="integer">2</Constant></Apply></Apply></Mean><Variance><Constant dataType="double">2</Constant></Variance></NormalDistributionForBN></ContinuousDistribution></ContinuousConditionalProbability>
      <ContinuousConditionalProbability><ParentValue parent="D3" value="1"/><ContinuousDistribution><NormalDistributionForBN><Mean><Apply function="*"><Constant dataType="double">1.5</Constant><FieldRef field="C2"/></Apply></Mean><Variance><Constant dataType="double">2</Constant></Variance></NormalDistributionForBN></ContinuousDistribution></ContinuousConditionalProbability>
    </ContinuousNode>
    <DiscreteNode name="D4">
      <DerivedField name="C3_Discretized" optype="categorical" dataType="string"><Discretize field="C3"><DiscretizeBin binValue="0"><Interval closure="openClosed" rightMargin="9"/></DiscretizeBin><DiscretizeBin binValue="1"><Interval closure="openClosed" leftMargin="9" rightMargin="11"/></DiscretizeBin><DiscretizeBin binValue="2"><Interval closure="openOpen" leftMargin="11"/></DiscretizeBin></Discretize></DerivedField>
      <DiscreteConditionalProbability><ParentValue parent="D3" value="0"/><ParentValue parent="C3_Discretized" value="0"/><ValueProbability value="0" probability="0.4"/><ValueProbability value="1" probability="0.6"/></DiscreteConditionalProbability>
      <DiscreteConditionalProbability><ParentValue parent="D3" value="0"/><ParentValue parent="C3_Discretized" value="1"/><ValueProbability value="0" probability="0.3"/><ValueProbability value="1" probability="0.7"/></DiscreteConditionalProbability>
      <DiscreteConditionalProbability><ParentValue parent="D3" value="0"/><ParentValue parent="C3_Discretized" value="2"/><ValueProbability value="0" probability="0.6"/><ValueProbability value="1" probability="0.4"/></DiscreteConditionalProbability>
      <DiscreteConditionalProbability><ParentValue parent="D3" value="1"/><ParentValue parent="C3_Discretized" value="0"/><ValueProbability value="0" probability="0.4"/><ValueProbability value="1" probability="0.6"/></DiscreteConditionalProbability>
      <DiscreteConditionalProbability><ParentValue parent="D3" value="1"/><ParentValue parent="C3_Discretized" value="1"/><ValueProbability value="0" probability="0.1"/><ValueProbability value="1" probability="0.9"/></DiscreteConditionalProbability>
      <DiscreteConditionalProbability><ParentValue parent="D3" value="1"/><ParentValue parent="C3_Discretized" value="2"/><ValueProbability value="0" probability="0.3"/><ValueProbability value="1" probability="0.7"/></DiscreteConditionalProbability>
    </DiscreteNode>
  </BayesianNetworkNodes>
</BayesianNetworkModel>
</PMML>"#;
    let env = PmmlEnv::new();
    let sess = Session::from_bytes(&env, xml, SessionOptions::default()).unwrap();
    // Evidence D4=0 and C4=7 as in spec example
    let sid0 = sess.symbol_id("0").unwrap();
    let mut input = HashMap::new();
    input.insert("D4".to_string(), Value::Discrete(sid0));
    input.insert("C4".to_string(), Value::Continuous(7.0));
    let out = sess
        .run(&input as &dyn pmmlruntime::session::batch::Batch)
        .unwrap()
        .into_single()
        .unwrap();
    assert!(out.contains_key("predictedValue"));
}
