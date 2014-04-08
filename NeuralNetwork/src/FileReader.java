import java.io.BufferedReader;
import java.io.FileInputStream;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.OutputStreamWriter;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.Collections;

public class FileReader {
	public static BufferedReader inStream;
	public static PrintWriter outStream;
	
	public static String[][] allData = new String[209][6];
	public static ArrayList<NeuralNetwork> networksList = new ArrayList<NeuralNetwork>();
		
	public static ArrayList<NeuralNetwork> readAllData() throws IOException {
		String oneLine = null;
		oneLine = inStream.readLine();
		//Get max values for each of the attributes so that you can normalize each Neuron
		double[] maxValues = new double[10];
		//Read in all the data into one array
		for(int i = 0; i<209; i++){
			allData[i] = oneLine.split(",");
			for(int j = 2; j<10; j++){
				double current = Integer.parseInt(""+allData[i][j]);
				if(current>maxValues[j]){
					maxValues[j] = current;
				}
			}
			oneLine = inStream.readLine();
		}
		for(int i = 0; i<209; i++){	
			//ignore the first two elements
			Neuron a = new Neuron();
			a.a = Integer.parseInt(allData[i][2])/maxValues[2];
	
			Neuron b = new Neuron();
			b.a = Integer.parseInt(allData[i][3])/maxValues[3];
		
			Neuron c = new Neuron();
			c.a = Integer.parseInt(allData[i][4])/maxValues[4];
			
			Neuron d = new Neuron();
			d.a = Integer.parseInt(allData[i][5])/maxValues[5];
			
			Neuron e = new Neuron();
			e.a = Integer.parseInt(allData[i][6])/maxValues[6];
			
			Neuron f = new Neuron();
			f.a = Integer.parseInt(allData[i][7])/maxValues[7];
			
			NeuralNetwork n = new NeuralNetwork();
			n.inputLayer[0]=a;
			n.inputLayer[1]=b;
			n.inputLayer[2]=c;
			n.inputLayer[3]=d;
			n.inputLayer[4]=e;
			n.inputLayer[5]=f;
			
			networksList.add(n);
			
			
		}

		
		//Then shuffle data points and distribute into 5 sets, 4 for training and 1 for testing
		//represented by arrays.
		Collections.shuffle(networksList);
		return networksList;
	}
		
		//BELOW ARE I/O METHODS
		
		/**
		 * Sets the input and output files as inStream and outStream, respectively
		 *
		 * @param infile	Input file
		 * @param outfile	Output file
		 */
		public static void setInputOuput(String inFile, String outFile)
		{
			try
			{
				FileInputStream fis = new FileInputStream(inFile); 
				InputStreamReader in = new InputStreamReader(fis, "UTF-8");
				inStream = new BufferedReader(in);
				
				FileOutputStream fos = new FileOutputStream(outFile); 
				OutputStreamWriter out = new OutputStreamWriter(fos, "UTF-8");
				outStream = new PrintWriter(out);
			}
			catch(FileNotFoundException e)
			{
				e.printStackTrace();
			}
			catch(IOException e)
			{
				e.printStackTrace();
			}
		}
	
	/**
	 * Prints string to outStream
	 *
	 * @param s		The string to print
	 */
	public static void display(String s)
	{
		outStream.print(s);
	}
	
	/**
	 * Prints string to outStream and moves to next line
	 *
	 * @param s		The string to print
	 */
	public static void displayln(String s)
	{
		outStream.println(s);
	}
	
	/**
	 * Closes the I/O streams
	 *
	 */
	public static void closeIO()
	{
		try
		{
			inStream.close();
			outStream.close();
		}
		catch(IOException e)
		{
			e.printStackTrace();
		}
	} // end closeIO

		
		
}//end FileReader class
