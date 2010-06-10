package edu.uoregon.cs.apriori;

import java.io.FileNotFoundException;
import java.io.PrintStream;
import java.util.*;

import weka.associations.ItemSet;
import weka.core.FastVector;
import weka.core.Instances;

public class RuleParser {
	private static List<FastVector[]> ruleSets = new ArrayList<FastVector[]>();
	private static Instances attributes = null;
	private static LinkedHashMap<String,Integer> ruleMatches = new LinkedHashMap<String, Integer>();
	private static String aprioriRunSummaries = "";
	
	public static void setAttributes(Instances attributes) {
		RuleParser.attributes = attributes;
	}
	
	/**
	 * Adds a ruleset to the list of rulesets.
	 * @param rules
	 * 		A FastVector[] containing association rules from an Apriori execution
	 */
	public static void addRules(FastVector[] rules) {
		synchronized (ruleSets) {
			ruleSets.add(rules);
		}
	}
	
	public static void addAprioriRunSummary(String summary) {
		synchronized (aprioriRunSummaries) {
			aprioriRunSummaries += summary + "\n\n";
		}
	}
	
	/**
	 * Matches all rules against each other and counts the number of rule matches 
	 * between the different results.
	 */
	public static void ruleMatcher() {
		HashMap<String, Integer> tempMatches = new HashMap<String, Integer>();
		
		for (int i = 0; i < ruleSets.size()-1; i++) {
			for (int j = i+1; j < ruleSets.size(); j++) {
				FastVector[] fvi = ruleSets.get(i); // Contains rules for the ith run
				FastVector[] fvj = ruleSets.get(j); // Contains rules for the jth run
				
				int LEFT = 0;
				int RIGHT = 1;
				// fvi[0] and fvj[0] correspond to a vector of left-hand side itemsets
				// for the ith and jth run respectively
				
				// Match left and right-hand sides
				int numItemsetsFVI = fvi[LEFT].size();
				int numItemsetsFVJ = fvj[LEFT].size();
				for (int k = 0; k < numItemsetsFVI; k++) {
					for (int l = 0; l < numItemsetsFVJ; l++) {
						boolean match = true;
						if (!fvi[LEFT].elementAt(k).equals(fvj[LEFT].elementAt(l)))
							match = false;
						
						if (match && !fvi[RIGHT].elementAt(k).equals(fvj[RIGHT].elementAt(l)))
							match = false;
						
						if (match) {
							String rule = toRuleString((ItemSet)fvi[LEFT].elementAt(k)) + 
							" ==> " + toRuleString((ItemSet)fvi[RIGHT].elementAt(k));
							if (tempMatches.containsKey(rule))
								tempMatches.put(rule, tempMatches.get(rule) + 1);
							else
								tempMatches.put(rule, 1);
						}
					}
				}
			}
		}
		
		// Sort the matching rules according to the number of matches
		// Inspired by the StackOverflow thread:
		// http://stackoverflow.com/questions/109383/how-to-sort-a-mapkey-value-on-the-values-in-java
		List<Map.Entry<String, Integer>> list = new LinkedList<Map.Entry<String, Integer>>(tempMatches.entrySet());
		Collections.sort(list, new Comparator<Map.Entry<String, Integer>>() {
			@Override
			public int compare(Map.Entry<String, Integer> o1,
					Map.Entry<String, Integer> o2) {
				return o2.getValue().compareTo(o1.getValue());
			}
			});
		
		// Put the sorted rules into to hashmap
		for (Map.Entry<String, Integer> entry : list)
			ruleMatches.put(entry.getKey(), entry.getValue());
	}
	
	/**
	 * Create a String representation of an Apriori itemset
	 * @param is
	 * 		The Apriori itemset
	 * @return
	 * 		A String
	 */
	private static String toRuleString(ItemSet is) {
		String res = "";
		int car = attributes.classIndex();
		
		// If class association rule mining is on
		// The right hand side itemset will only have information on the class attribute
		// and therefore only contain one element
		if (car >= 0 && is.items().length == 1) {
			res += attributes.attribute(car).name()+'=';
			res += attributes.attribute(car).value(is.itemAt(0)) + " ";
		}
		else {
			for (int i = 0; i < attributes.numAttributes(); i++) {
				// Skip the class index
				if (i == car)
					continue;

				int itemSetIndex = i;
				if (car >= 0 && i > attributes.classIndex()) {
					itemSetIndex--;
				}

				if (is.itemAt(itemSetIndex) != -1) {
					res += attributes.attribute(i).name()+'=';
					res += attributes.attribute(i).value(is.itemAt(itemSetIndex)) + " ";
				}
			}
		}
		return res;
	}
	
	public static void printSummaryToFile() {
		try {
			PrintStream ps = new PrintStream("rulematches");
			for (String s : ruleMatches.keySet()) {
				ps.println(ruleMatches.get(s) + ": " + s);
			}
			ps.close();
			ps = new PrintStream("rules");
			ps.print(aprioriRunSummaries);
			ps.close();
		} catch (FileNotFoundException e) {
			e.printStackTrace();
		}
	}
}
