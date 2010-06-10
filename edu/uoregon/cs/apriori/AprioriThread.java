package edu.uoregon.cs.apriori;

import weka.associations.Apriori;
import weka.associations.AssociatorEvaluation;

/**
 * A Runnable class that simply runs Weka's Apriori algorithm
 */
public class AprioriThread implements Runnable {
	Apriori apriori = null;
	String[] options;
	
	public AprioriThread(String[] options) {
		apriori = new Apriori();
		this.options = options;
	}
	
	/**
	 * Run method.
	 * 1. Runs Apriori with the given options
	 * 2. Prints the found rules
	 * 3. Adds the rules to the RuleParser
	 */
	public void run() {
		try {
			// Run the Apriori algorithm
			AssociatorEvaluation.evaluate(apriori, options);
			
			// Print rules as they finish and also store them in the RuleParser
			System.out.println(apriori);
			RuleParser.addAprioriRunSummary(apriori.toString());
			
			// Add the rules to the RuleParser
			RuleParser.addRules(apriori.getAllTheRules());
		} catch (Exception e) {
			e.printStackTrace();
		}
	}
}
