


/*
  IMU Classifier

  This example uses the on-board IMU to start reading acceleration and gyroscope
  data from on-board IMU, once enough samples are read, it then uses a
  TensorFlow Lite (Micro) model to try to classify the movement as a known gesture.

  Note: The direct use of C/C++ pointers, namespaces, and dynamic memory is generally
        discouraged in Arduino examples, and in the future the TensorFlowLite library
        might change to make the sketch simpler.

  The circuit:
  - Arduino Nano 33 BLE or Arduino Nano 33 BLE Sense board.

  Created by Don Coleman, Sandeep Mistry
  Modified by Dominic Pajak, Sandeep Mistry

  This example code is in the public domain.
*/

// #include <Arduino_LSM9DS1.h>

#include <TensorFlowLite.h>
// #include <tensorflow/lite/micro/all_ops_resolver.h>
#include <tensorflow/lite/micro/micro_mutable_op_resolver.h>
#include <tensorflow/lite/micro/micro_interpreter.h>
#include <tensorflow/lite/schema/schema_generated.h>
// #include <tensorflow/lite/micro/micro_error_reporter.h>
// #include <tensorflow/lite/version.h>

// // #include "model.h"

#include "finalModel-2B.h"
#include "paper2.h"

const float accelerationThreshold = 2.5; // threshold of significant in G's
const int numSamples = 119;

int samplesRead = numSamples;

// global variables used for TensorFlow Lite (Micro)
// tflite::MicroErrorReporter tflErrorReporter;

// pull in all the TFLM ops, you can remove this line and
// only pull in the TFLM ops you need, if would like to reduce
// the compiled size of the sketch.
// tflite::AllOpsResolver tflOpsResolver;
// tflite::MicroMutableOpResolver<6> tflOpsResolver;
tflite::MicroMutableOpResolver<7> tflOpsResolver;

const tflite::Model* tflModel = nullptr;
tflite::MicroInterpreter* tflInterpreter = nullptr;
TfLiteTensor* tflInputTensor = nullptr;
TfLiteTensor* tflOutputTensor = nullptr;

// Create a static memory buffer for TFLM, the size may need to
// be adjusted based on the model you are using
constexpr int tensorArenaSize = 128 * 1024;
byte tensorArena[tensorArenaSize] __attribute__((aligned(16)));

// array to map gesture index to a name
const char* CLASSES[] = {
  "rock",
  "paper",
  "scissors"
};

#define NUM_GESTURES (sizeof(CLASSES) / sizeof(CLASSES[0]))

void setup() {
  Serial.begin(9600);
  while (!Serial);

  // // initialize the IMU
  // if (!IMU.begin()) {
  //   Serial.println("Failed to initialize IMU!");
  //   while (1);
  // }

  // // print out the samples rates of the IMUs
  // Serial.print("Accelerometer sample rate = ");
  // Serial.print(IMU.accelerationSampleRate());
  // Serial.println(" Hz");
  // Serial.print("Gyroscope sample rate = ");
  // Serial.print(IMU.gyroscopeSampleRate());
  // Serial.println(" Hz");

  // Serial.println();

  // get the TFL representation of the model byte array
  // tflModel = tflite::GetModel(g_model);
  tflModel = tflite::GetModel(model);
  if (tflModel->version() != TFLITE_SCHEMA_VERSION) {
    Serial.println("Model schema mismatch!");
    while (1);
  }

  tflOpsResolver.AddConv2D();
  tflOpsResolver.AddMaxPool2D();
  tflOpsResolver.AddFullyConnected();
  tflOpsResolver.AddReshape();
  tflOpsResolver.AddSoftmax();
  tflOpsResolver.AddQuantize();
  tflOpsResolver.AddDequantize();

  // Create an interpreter to run the model
  tflInterpreter = new tflite::MicroInterpreter(tflModel, tflOpsResolver, tensorArena, tensorArenaSize);

  // Allocate memory for the model's input and output tensors
  // tflInterpreter->AllocateTensors();
  TfLiteStatus allocateStatus = tflInterpreter->AllocateTensors();

  if (allocateStatus != kTfLiteOk) {
    Serial.println("AllocateTensors failed!");
    while (1);
  }

  // Get pointers for the model's input and output tensors
  tflInputTensor = tflInterpreter->input(0);
  tflOutputTensor = tflInterpreter->output(0);
}

void loop() {
  int input_len = tflInputTensor->bytes / sizeof(float);

  for (int i = 0; i < input_len; i++) {
    tflInputTensor->data.f[i] = image_data[i] / 255.0f;
  }

  TfLiteStatus invokeStatus = tflInterpreter->Invoke();
  if (invokeStatus != kTfLiteOk) {
    Serial.println("Invoke failed!");
    return;
  }

  // f_t rock_score = tflOutputTensor->data.f[0];
  // f_t paper_score = tflOutputTensor->data.f[1];
  // f_t scissors_score = tflOutputTensor->data.f[2];
  float rock_score = tflOutputTensor->data.f[0];
  float paper_score = tflOutputTensor->data.f[1];
  float scissors_score = tflOutputTensor->data.f[2];

  Serial.print("rock: "); Serial.print(rock_score);
  Serial.print(" paper: "); Serial.print(paper_score);
  Serial.print(" scissors: "); Serial.println(scissors_score);

  // uint8_t max_score = rock_score;
  float max_score = rock_score;
  int max_index = 0;
  if (paper_score > max_score) { max_score = paper_score; max_index = 1; }
  if (scissors_score > max_score) { max_score = scissors_score; max_index = 2; }
  
  Serial.print("Prediction: ");
  Serial.println(CLASSES[max_index]);

  while (true) {
    delay(10000);
  }
}
