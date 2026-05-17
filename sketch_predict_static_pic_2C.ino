
#include <TensorFlowLite.h>
#include <tensorflow/lite/micro/micro_mutable_op_resolver.h>
#include <tensorflow/lite/micro/micro_interpreter.h>
#include <tensorflow/lite/schema/schema_generated.h>

#include "finalModel_1B_2C.h"
#include "image_data.h"

tflite::MicroMutableOpResolver<7> tflOpsResolver;

const tflite::Model* tflModel = nullptr;
tflite::MicroInterpreter* tflInterpreter = nullptr;
TfLiteTensor* tflInputTensor = nullptr;
TfLiteTensor* tflOutputTensor = nullptr;

constexpr int tensorArenaSize = 64 * 1024;
byte tensorArena[tensorArenaSize] __attribute__((aligned(16)));

const char* CLASSES[] = {
  "rock",
  "paper",
  "scissors"
};

void setup() {
  Serial.begin(9600);
  while (!Serial);

  Serial.println("Starting...");

  tflOpsResolver.AddConv2D();
  tflOpsResolver.AddMaxPool2D();
  tflOpsResolver.AddFullyConnected();
  tflOpsResolver.AddReshape();
  tflOpsResolver.AddSoftmax();
  tflOpsResolver.AddQuantize();
  tflOpsResolver.AddDequantize();

  tflModel = tflite::GetModel(model);

  if (tflModel->version() != TFLITE_SCHEMA_VERSION) {
    Serial.println("Model schema mismatch!");
    while (1);
  }

  tflInterpreter = new tflite::MicroInterpreter(
    tflModel,
    tflOpsResolver,
    tensorArena,
    tensorArenaSize
  );

  TfLiteStatus allocateStatus = tflInterpreter->AllocateTensors();

  if (allocateStatus != kTfLiteOk) {
    Serial.println("AllocateTensors failed!");
    while (1);
  }

  tflInputTensor = tflInterpreter->input(0);
  tflOutputTensor = tflInterpreter->output(0);

  Serial.println("Tensors allocated.");
  Serial.print("Input type: ");
  Serial.println(tflInputTensor->type);
  Serial.print("Output type: ");
  Serial.println(tflOutputTensor->type);
}

void loop() {
  float input_scale = tflInputTensor->params.scale;
  int input_zero_point = tflInputTensor->params.zero_point;

  for (int i = 0; i < tflInputTensor->bytes; i++) {
    float normalized = image_data[i] / 255.0f;
    int quantized = round(normalized / input_scale) + input_zero_point;

    if (quantized < -128) quantized = -128;
    if (quantized > 127) quantized = 127;

    tflInputTensor->data.int8[i] = (int8_t)quantized;
  }

  TfLiteStatus invokeStatus = tflInterpreter->Invoke();

  if (invokeStatus != kTfLiteOk) {
    Serial.println("Invoke failed!");
    return;
  }

  int8_t rock_q = tflOutputTensor->data.int8[0];
  int8_t paper_q = tflOutputTensor->data.int8[1];
  int8_t scissors_q = tflOutputTensor->data.int8[2];

  float output_scale = tflOutputTensor->params.scale;
  int output_zero_point = tflOutputTensor->params.zero_point;

  float rock = (rock_q - output_zero_point) * output_scale;
  float paper = (paper_q - output_zero_point) * output_scale;
  float scissors = (scissors_q - output_zero_point) * output_scale;

  Serial.print("Raw int8 -> ");
  Serial.print("rock: "); Serial.print(rock_q);
  Serial.print(" paper: "); Serial.print(paper_q);
  Serial.print(" scissors: "); Serial.println(scissors_q);

  Serial.print("Scores -> ");
  Serial.print("rock: "); Serial.print(rock, 4);
  Serial.print(" paper: "); Serial.print(paper, 4);
  Serial.print(" scissors: "); Serial.println(scissors, 4);

  float max_score = rock;
  int max_index = 0;

  if (paper > max_score) {
    max_score = paper;
    max_index = 1;
  }

  if (scissors > max_score) {
    max_score = scissors;
    max_index = 2;
  }

  Serial.print("Prediction: ");
  Serial.println(CLASSES[max_index]);

  while (true) {
    delay(10000);
  }
}
