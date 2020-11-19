package moa;

import com.yahoo.labs.samoa.instances.Instance;

public class DistanceInstance {

	private double distance;
	private Instance instancia;
	
	public DistanceInstance(double distance, Instance instancia) {
		super();
		this.distance = distance;
		this.instancia = instancia;
	}
	
	public DistanceInstance() {
		super();
	}

	public double getDistance() {
		return distance;
	}

	public void setDistance(double distance) {
		this.distance = distance;
	}

	public Instance getInstancia() {
		return instancia;
	}

	public void setInstancia(Instance instancia) {
		this.instancia = instancia;
	}
	
	
	
}
