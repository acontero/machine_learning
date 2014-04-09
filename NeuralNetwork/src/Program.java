import java.io.IOException;
import java.util.ArrayList;
import java.util.Random;


public class Program {
	public static double[][] inputWeights = new double [6][3];
	public static double[][] hiddenWeights = new double [3][1];
	public static double[][] oldInputWeights = new double[6][3];
	public static double[][] oldHiddenWeights = new double [3][1];
	
	static double learningRate = 0.5; //learning rate is between 0 and 1
	static double errorThreshold = 0.1;
//	static double errorThreshold = 0.01;
//	static double errorThreshold = 0.001;

	public static void main(String argz[]) throws IOException{
		FileReader.setInputOuput(argz[0], argz[1]);
		//networksList is a list of the 209 networks
		ArrayList<NeuralNetwork> networksList = FileReader.readAllData();
		//networks an array of all 209 networks
		NeuralNetwork[] networks = new NeuralNetwork[networksList.size()];
		networksList.toArray(networks);
		int dataCount = networksList.size();
		
		//perform back propagation algorithm on the beginning 4/5 of the networksList.
		ArrayList<NeuralNetwork> currentNetworkList = new ArrayList<NeuralNetwork>();
		int end = networksList.size() - (networksList.size()/5);
		for(int j = 0; j<end; j++){
			currentNetworkList.add(networks[j]);
		}
		//return from backPropogation a set of networks that have been modeled
		currentNetworkList = backPropagation(currentNetworkList);
		
		FileReader.displayln("*************************");
		FileReader.displayln("*************************");
		FileReader.displayln("*************************");
		FileReader.displayln("*************************");
		FileReader.displayln("*************************");
		FileReader.displayln("*************************");
		FileReader.displayln("*************************");
		FileReader.displayln("*************************");
		FileReader.displayln("*************************");
		FileReader.displayln("*************************");
		FileReader.displayln("TESTING DATA: RESULTS:");
		
		//run testing code on the remaining 1/5 of the networksList
		ArrayList<NeuralNetwork> testNetworkList = new ArrayList<NeuralNetwork>();
		for(int j = end; j<networksList.size(); j++){
			testNetworkList.add(networks[j]);
		}
		testNetworkList = finalTest(testNetworkList);
		
		//STILL NEED TO IMPLEMENT THE REMAINDER OF THE 5-FOLD CROSS VALIDATION
		
		
		
		
		FileReader.closeIO();
	}
	
	public static ArrayList<NeuralNetwork> backPropagation(ArrayList<NeuralNetwork> c){
		//c is a list of networks
		int numIterations = 0;
		boolean weightsConverged = false;
		double MSE = 100;
		doRandomization();
		while(MSE>errorThreshold && numIterations<5000){
			//STEP 1: initialize weight matrixes in between each layer	
			double maxError = 0;
			double minError = 0;
			double totalErrorSum = 0;
			double errorSquaredSum = 0;
			//STEP 2: For each network in from input vector x, assign values to all neurons in input layer
			//NOTE: This was already done in FileReader when reading in and shuffling.
			//For each network in c, propagate calculations up
			int networkCount = 0;
			for (NeuralNetwork network : c) {
				int indexNeuronj = 0;
				//STEP 3: for each Neuron in the hidden layer calculate it's in and a
				for (Neuron j: network.hiddenLayer){
					j.in = getSum(indexNeuronj, inputWeights, network.inputLayer);
					j.a = sigmoidActivationFunction(j.in);

					indexNeuronj++;
				}
				//STEP 4: Calculate output
				network.output.in = getOutputSum(0,hiddenWeights, network.hiddenLayer);
				network.output.a = sigmoidActivationFunction(network.output.in);
				
				//ERROR CALCULATIONS
				double outputError = Math.abs(network.inputLayer[4].a - network.output.a); //compare against published CPU performance which is at index 4
				totalErrorSum += outputError;
				errorSquaredSum += Math.pow(outputError,2);
				if(outputError > maxError) maxError = outputError;
				if(outputError < minError) minError = outputError;
				//Display info about current error
//				FileReader.displayln("Expected output: " + network.inputLayer[4].a);
//				FileReader.displayln("Actual output for this line of data: " + network.output.a);
//				FileReader.displayln("Current Output Error: " + outputError);
//				FileReader.displayln("");
				
				/* Propagate deltas backward from output layer to input layer */
				double[] deltaOutput = new double[1]; 
				deltaOutput[0] = derivativeActivationFunction(network.output.in)*outputError;
				double[] deltaj = new double[network.hiddenLayer.length];
				int j_index = 0;
				for (Neuron j: network.hiddenLayer){
					deltaj[j_index] = derivativeActivationFunction(j.in) * getDeltaSum(j_index,hiddenWeights,deltaOutput);
					j_index++;
				}

				double[] deltai = new double[network.inputLayer.length];
				int i_index = 0;
				for (Neuron i: network.inputLayer){
					deltai[i_index] = derivativeActivationFunction(i.in) * getDeltaSum(i_index,inputWeights,deltaj);
					i_index++;
				}
				
				/* Update every weight in the network using deltas */
				i_index = 0;
				for (Neuron i: network.inputLayer){
					for(int x = 0; x<3; x++){
						inputWeights[i_index][x] = inputWeights[i_index][x] + (learningRate * i.a * deltaj[x]);
					}
					i_index++;
				}
				j_index = 0;
				for (Neuron j: network.hiddenLayer){
					hiddenWeights[j_index][0] = hiddenWeights[j_index][0] + (learningRate * j.a * deltaOutput[0]);
					j_index++;
				}
				networkCount++;
			}//finished with all networks in this division
			
			//ERROR CALCULATIONS FOR ENTIRE ITERATION:
			FileReader.displayln("ITERATION SUMMARY: ");
			FileReader.displayln("IterationNumber: " + numIterations);
			FileReader.displayln("AVG error difference for 209 networks: " + totalErrorSum/c.size());
			MSE = errorSquaredSum/(2*c.size());
			FileReader.displayln("Mean Squared Error for 209 networks: " + MSE);
			FileReader.displayln("MAX error difference for 209 networks: " + maxError);
			FileReader.displayln("MIN error difference for 209 networks: " + minError);
			FileReader.displayln(" ");

			//check if weights have changed since last iteration (test for convergence)
			weightsConverged = checkWeightDifference(oldInputWeights,inputWeights,oldHiddenWeights,hiddenWeights);
			if(weightsConverged==true){
				FileReader.displayln("NOTE: Weights have converged to less than 0.01!");
				FileReader.displayln("");
			}
			
			//if model has not converged and if totalError is not less than threshold, redo whole thing with new randomized weights
			//keep going until reach max number of iterations.
			oldInputWeights = copyOver(inputWeights);
			oldHiddenWeights = copyOver(hiddenWeights);
			numIterations++;
			
		}//done going through while loop
	
		FileReader.displayln("Done Training!");
		FileReader.displayln("In iteration: " + numIterations);
		FileReader.displayln(" ");
		FileReader.displayln(" ");
		return c;
	}
	
	public static void doRandomization(){
		randomizeWeights(inputWeights);
		randomizeWeights(hiddenWeights);
	}
	
	public static ArrayList<NeuralNetwork> finalTest(ArrayList<NeuralNetwork> c){
		double entireError = 0;
		int networkCount = 0;
		for (NeuralNetwork network : c) {
			int indexNeuronj = 0;
			//For each Neuron in the hidden layer calculate it's in and a
			for (Neuron j: network.hiddenLayer){
				j.in = getSum(indexNeuronj, inputWeights, network.inputLayer);
				j.a = sigmoidActivationFunction(j.in);

				indexNeuronj++;
			}
			//Calculate output
			network.output.in = getOutputSum(0,hiddenWeights, network.hiddenLayer);
			network.output.a = sigmoidActivationFunction(network.output.in);
			
			//PRINT INFO. OUT
			FileReader.displayln("Expected output: " + network.inputLayer[4].a);
			FileReader.displayln("Actual output for this line of data: " + network.output.a);
			double outputError = Math.abs(network.inputLayer[4].a - network.output.a); //compare against published CPU performance which is at index 4
			FileReader.displayln("Current Output Error: " + outputError);
			entireError += outputError;
			FileReader.displayln("");
		}
		FileReader.displayln("TOTAL testing output error = " + entireError);
		FileReader.displayln("Which is an avg of: " + entireError/c.size() + " percent error.");	
		
		return c;
	}
	
	public static void printDelta(double[] a){
		for(int i = 0; i<a.length; i++){
			FileReader.displayln(""+a[i]);
		}
	}
	
	public static double[][] copyOver(double[][] a){
		double[][] n = new double[a.length][a[0].length];
		
		for(int i = 0; i<a.length; i++){
			for(int j = 0; j<a[i].length;j++){
				n[i][j] = a[i][j];
			}
		}
		
		return n;
	}
	public static boolean checkWeightDifference(double[][] oi, double[][]i, double[][] oh, double[][] h){
		double weightThreshold = 0.01;
		//max difference between old and current
		double maxDiffInput = 0;
		for(int x = 0; x<i.length; x++){
			for(int y = 0; y < i[x].length; y++){
//				System.out.println("x,y,value: " + x + y + oi[x][y] + i[x][y]);
				double diff = Math.abs(oi[x][y] - i[x][y]);
				if(diff>maxDiffInput){
//					FileReader.displayln("new maxDiffHidden!");
					maxDiffInput = diff;
				}
				if(maxDiffInput > weightThreshold){
//					FileReader.displayln("maxDiff in input = " + maxDiffInput);
					return false;
				}
			}
		}
//		FileReader.displayln("maxDiff in input = " + maxDiffInput);
	
		double maxDiffHidden = 0;
		for(int x = 0; x<h.length; x++){
			for(int y = 0; y < h[x].length; y++){
//				System.out.println("x,y,value: " + x + y + oh[x][y] + h[x][y]);
				double diff = Math.abs(oh[x][y] - h[x][y]);
				if(diff>maxDiffHidden){
//					FileReader.displayln("new maxDiffHidden!");
					maxDiffHidden = diff;
				}
				if(maxDiffHidden > weightThreshold){
//					FileReader.displayln("maxDiff in hidden = " + maxDiffHidden);
					return false;
				}
			}
		}
//		FileReader.displayln("maxDiff in hidden = " + maxDiffHidden);
		
		if(maxDiffInput > weightThreshold || maxDiffInput > weightThreshold) return false;
		
		return true;
	}
	
	public static double getDeltaSum(int neuronIndex, double[][] weights, double[] delta){
		double sum = 0;
		for (int i = 0; i<weights[neuronIndex].length; i++){
			for (int j = 0; j<delta.length; j++){
				sum += weights[neuronIndex][i] * delta[j];
			}
		}
		return sum;
	}
	
	public static double getSum(int indexNeuronj,double[][] inputWeights, Neuron[]inputLayer){
		double sum = 0;
		//length is 6 and we only want the first 4 values so we go until length-2
		for(int i = 0 ; i<inputWeights.length-2; i++){
			sum += inputWeights[i][indexNeuronj] * inputLayer[i].a;
		}
		return sum;
	}
	
	public static double getOutputSum(int indexNeuron, double[][] hiddenWeights, Neuron[]hiddenLayer){
		double sum = 0;
		for(int i = 0 ; i<hiddenWeights[indexNeuron].length; i++){
			sum += hiddenWeights[i][indexNeuron] * hiddenLayer[i].a;
		}
		return sum;
	}
	
	public static void randomizeWeights(double[][] matrix){
		//initialize randomWeight between 0 and +2.0
		Random rnd = new Random();
		double minWeight = 0.0;
		double maxWeight = 1.0;

		//initializing weights between input layer and hidden layer
		for (int i = 0; i < matrix.length; i++){
			for (int j = 0; j < matrix[i].length; j++){
				matrix [i][j] = minWeight + (maxWeight - minWeight)* rnd.nextDouble();
			}
		}
	}
	
	/**
	 * Calculates the output value a for a Neuron node by passing weighted sum of inputs, 
	 * in, through the activation function:
	 * sigmoid activation function: 1/(1+e-x)
	 * */
	public static double sigmoidActivationFunction(double in){
		return 1.0/(1.0 + Math.exp(-in));
	}
	
	public static void printWeights(double[][] a){
		for (int i = 0; i<a.length; i++){
			for (int j = 0; j<a[i].length; j++){
				FileReader.display(a[i][j] + " ");
			}
			FileReader.displayln("");
		}
		FileReader.displayln("");
	}
	
	public static double derivativeActivationFunction(double o){
		double answer = o * (1-o); 
		return answer;
	}
}
