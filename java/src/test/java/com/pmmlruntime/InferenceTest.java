package com.pmmlruntime;

import org.junit.Test;
import static org.junit.Assert.*;
import java.io.File;
import java.util.*;

public class InferenceTest {
    private static String findModel() {
        String userDir = System.getProperty("user.dir", ".");
        String[] candidates = {
            "bench/pmml/DecisionTreeIris.pmml",
            "../bench/pmml/DecisionTreeIris.pmml",
            "../../bench/pmml/DecisionTreeIris.pmml",
            userDir + "/bench/pmml/DecisionTreeIris.pmml",
            userDir + "/../bench/pmml/DecisionTreeIris.pmml",
            userDir + "/../../bench/pmml/DecisionTreeIris.pmml"
        };
        for (String c : candidates) {
            if (new File(c).exists()) return new File(c).getAbsolutePath();
        }
        // Default: let native layer report the missing file.
        return "bench/pmml/DecisionTreeIris.pmml";
    }

    @Test public void scoresIris() throws Exception {
        try (PmmlEnv env = PmmlEnv.create()) {
            try (PmmlSession s = PmmlSession.fromFile(env, findModel())) {
                Map<String,Object> in = new HashMap<>();
                in.put("Petal.Length", 1.4); in.put("Petal.Width", 0.2);
                Map<String,Object> out = s.run(in);
                assertTrue("expected predictedValue in " + out, out.containsKey("predictedValue"));
            }
        }
    }
}
