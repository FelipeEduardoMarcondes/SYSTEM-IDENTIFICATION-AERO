float y_hist[2] = {0.0, 0.0}; // y(k-1), y(k-2)
float u_hist[5] = {0.0, 0.0, 0.0, 0.0, 0.0}; // u(k-1) to u(k-5)

void setup() {
  Serial.begin(115200);
  while (!Serial) {
    ; // Wait for serial port to connect
  }
}

void loop() {
  if (Serial.available() > 0) {
    // Read the control signal u from Serial
    String input = Serial.readStringUntil('\n');
    input.trim();
    if (input.length() == 0) return;
    
    float u_current = input.toFloat();

    // Shift u history: u(k-5) = u(k-4), etc.
    for (int i = 4; i > 0; i--) {
      u_hist[i] = u_hist[i-1];
    }
    u_hist[0] = u_current;

    // Calculate y_k based on NARX polynomial extracted from Python
    // y_k = 1.9373*y(k-1) - 0.9545*y(k-2) 
    //       + 0.0007*u(k-2)^2 
    //       + 0.0000*u(k-4)*u(k-5)^2 
    //       - 0.0004*u(k-2)*u(k-3)
    
    float y_k = 1.9373 * y_hist[0] 
              - 0.9545 * y_hist[1] 
              + 0.0007 * (u_hist[1] * u_hist[1]) 
              + 0.0000 * (u_hist[3] * u_hist[4] * u_hist[4]) 
              - 0.0004 * (u_hist[1] * u_hist[2]);

    // Send y_k back to PC
    Serial.println(y_k, 6);

    // Shift y history
    y_hist[1] = y_hist[0];
    y_hist[0] = y_k;
  }
}
