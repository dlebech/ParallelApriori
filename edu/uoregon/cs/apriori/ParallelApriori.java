package edu.uoregon.cs.apriori;

import java.io.File;
import java.io.FileNotFoundException;
import java.util.ArrayList;
import java.util.Enumeration;
import java.util.concurrent.Executors;
import java.util.concurrent.ThreadPoolExecutor;
import java.util.concurrent.TimeUnit;

import weka.core.Attribute;
import weka.core.FastVector;
import weka.core.Instances;
import weka.core.converters.ArffLoader;
import weka.core.converters.ArffSaver;

/**
 * A class that implements a parallel execution scheme for the Apriori algorithm.
 * Apriori (from Weka) is run on samples of a given dataset.
 * The number of partitions are specified by the user.
 */
public class ParallelApriori {
	public static final int DEFAULT_NUM_THREADS = 2;
	public static final int DEFAULT_NUM_RULES_PER_THREAD = 10;
	private int numThreads = DEFAULT_NUM_THREADS;
	private int numRulesPerThread = DEFAULT_NUM_RULES_PER_THREAD;
	private int carIndex = -1; // Class association rule mining index, -1 is off
	private File f;
	
	public ParallelApriori(File f) throws Exception {
		this.f = f;
	}
	
	public void setNumThreads(int numThreads) {
		this.numThreads = numThreads;
	}
	
	public void setNumRulesPerThread(int numRulesPerThread) {
		this.numRulesPerThread = numRulesPerThread;
	}
	
	public void setCarIndex(int carIndex) {
		this.carIndex = carIndex;
	}
	
	/**
	 * Main algorithm
	 * 1. Reads in a dataset
	 * 2. Splits the dataset into several partitions
	 * 3. Runs the Apriori algorithm concurrently on each partition
	 * 4. Prints a summary of rules that overlap between executions.
	 * @throws Exception
	 */
	public void run() throws Exception {
		// Load the original arff file
		ArffLoader loader = new ArffLoader();
		loader.setFile(f);
		Instances structure = loader.getStructure();
		Instances[] partionedDataset = partitionInstances(loader.getDataSet(),numThreads);
		
		// Create several arff files
		String path = f.getParent();
		String[] pathsOfSubsets = new String[numThreads];
		
		System.out.println("Creating subset files");
		
		for (int i = 0; i < numThreads; i++) {
			pathsOfSubsets[i] = path + "/temp/" + i + ".arff"; 
			ArffSaver saver = new ArffSaver();
			saver.setStructure(structure);
			saver.setInstances(partionedDataset[i]);
			saver.setFile(new File(pathsOfSubsets[i]));
			saver.writeBatch();
		}
		
		System.out.println("File creation complete, starting Apriori threads");
	
		// Create and run apriori threads for each of these files
		ThreadPoolExecutor threadManager = (ThreadPoolExecutor) Executors.newFixedThreadPool(8);
		
		long start = System.currentTimeMillis();
		for (int i = 0; i < pathsOfSubsets.length; i++)
			threadManager.execute(createAprioriThread(pathsOfSubsets[i]));
		
		System.out.println("All threads started, waiting for results");
		
		// Wait for all threads to finish
		threadManager.shutdown();
		while (!threadManager.awaitTermination(5000, TimeUnit.MILLISECONDS))
			System.out.println("Still waiting");;
			
		// Calculate and print time of completion
		long time = System.currentTimeMillis()-start;
		
		System.out.println("Success! All threads finished, the whole thing took: " + time);
		
		// Let the rule parser know about the structure of the dataset
		// If we are mining class association rules, then the RuleParser needs to know this
		if (carIndex >= 0)
			structure.setClassIndex(carIndex-1); // Subtract 1 because structure is 0-indexed but the user input is not
		RuleParser.setAttributes(structure);
		
		// Report matches among rules from different executions
		RuleParser.ruleMatcher();
		
		// Print the rules
		RuleParser.printSummaryToFile();
	}
	
	/**
	 * Partitions a specific dataset into a number of subsets 
	 * @param instances
	 * @param num
	 * @return
	 * 		The new instances
	 */
	private Instances[] partitionInstances(Instances dataset, int num) {
		Instances[] res = new Instances[num];
		
		// Extract the attributes
		FastVector attr = new FastVector();
		Enumeration<Attribute> a = dataset.enumerateAttributes();
		while (a.hasMoreElements())
			attr.addElement(a.nextElement());
		
		int numInEach = dataset.numInstances()/num;
		
		// Create the subsamples
		for (int i = 0; i < num; i++) {
			Instances subset = new Instances("Subsample " + i + " from " + dataset.relationName(),attr,0);
			for (int j = 0; j < numInEach || (i == num-1 && (i*numInEach+j) < dataset.numInstances()); j++) {
				subset.add(dataset.instance(i*numInEach+j));
			}
			res[i] = subset;
		}
		return res;
	}
	
	/**
	 * Creates an AprioriThread object which can be used for a single run of Apriori.
	 * 
	 * Currently, only Arff files are supported and a number of settings for the Apriori
	 * algorithm have to be manually set in this method.
	 * @param pathToArff
	 * 		The path to the Arff dataset file
	 * @return
	 */
	private AprioriThread createAprioriThread(String pathToArff) {
		ArrayList<String> commands = new ArrayList<String>();
		commands.add("-t"); commands.add(pathToArff);
		commands.add("-N"); commands.add("" + numRulesPerThread);
		commands.add("-T"); commands.add("0"); 		// 0 = confidence, 1 = lift
		commands.add("-C"); commands.add("0.6"); 	// Min metric for -T
		commands.add("-D"); commands.add("0.001");	// Delta support
		commands.add("-M"); commands.add("0.001");	// Lower bound for minimum support
		commands.add("-U"); commands.add("0.1");	// Upper bound for minimum support
		
		if (carIndex >= 0) {
			// Mine class associate rules with the given index
			commands.add("-A"); commands.add("-c"); commands.add("" + carIndex);
		}
		
		return new AprioriThread(commands.toArray(new String[commands.size()]));
	}
	
	/**
	 * Main program. Takes up to four arguments.
	 * 1. The path to the Arff dataset file
	 * 2. The number of partitions for the dataset
	 * 3. The number of rules to generate per partition
	 * 4. The class association rule index. Blank or -1 is off 
	 * @param args
	 */
	public static void main(String[] args) {
		try {
			if (args.length >= 1) {
				File f = new File(args[0]);
				if (!f.exists())
					throw new FileNotFoundException();
				ParallelApriori pa = new ParallelApriori(f);
				if (args.length > 1)
					pa.setNumThreads(Integer.parseInt(args[1]));
				if (args.length > 2)
					pa.setNumRulesPerThread(Integer.parseInt(args[2]));
				if (args.length > 3)
					pa.setCarIndex(Integer.parseInt(args[3]));
				pa.run();
			}
		}
		catch (FileNotFoundException e) {
			e.printStackTrace();
		}
		catch (NumberFormatException e) {
			e.printStackTrace();
		}
		catch (Exception e) {
			e.printStackTrace();
		}
	}
}
