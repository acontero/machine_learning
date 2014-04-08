import java.util.ArrayList;

public class NeuralNetwork {
	public Neuron[] inputLayer = new Neuron[6]; //will have 5 neurons -- NOTE: the last two are expected outcomes
	public Neuron[] hiddenLayer = new Neuron[3]; //will have 3 neurons
	public Neuron output; //is only 1 neuron in our case
	
	public NeuralNetwork(){
		for (int i = 0; i<this.hiddenLayer.length; i++){
			Neuron n = new Neuron();
			this.hiddenLayer[i]=n;
		}
		Neuron o = new Neuron();
		this.output = o;	
	}
	
	public void printLayers(){
		for (Neuron i: this.inputLayer){
			System.out.print(i.in + "," + i.a + " ");
		}
		System.out.println("");
		for (Neuron j: this.hiddenLayer){
			System.out.print(j.in + "," + j.a + " ");
		}
		System.out.println("");
		System.out.print(this.output.in + "," + this.output.a + " ");
		System.out.println("");
	}
}
