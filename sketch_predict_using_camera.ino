/*
  Live Camera Gesture Classifier
  Arduino Nano 33 BLE + OV767X Camera
*/

#include <TensorFlowLite.h>
#include <tensorflow/lite/micro/micro_mutable_op_resolver.h>
#include <tensorflow/lite/micro/micro_interpreter.h>
#include <tensorflow/lite/schema/schema_generated.h>
#include <Arduino_OV767X.h> 

#include "finalModel_new.h"

// ==========================================
// 调试开关：取消注释下面这行开启“传图模式”，注释掉则为“预测模式”
// Debugging switch: Uncomment the line below to enable "Image Transfer Mode", comment it out to enable "Prediction Mode".
// #define DEBUG_MODE 
// ==========================================

// Camera native resolution (QCIF)
const int CAM_WIDTH = 176;
const int CAM_HEIGHT = 144;

const int MODEL_WIDTH = 32;  
const int MODEL_HEIGHT = 32; 
const int MODEL_PIXELS = MODEL_WIDTH * MODEL_HEIGHT;

unsigned short camera_buffer[CAM_WIDTH * CAM_HEIGHT]; 
uint8_t processed_image[MODEL_PIXELS]; 

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

// Preprocessing functions: center cropping + scaling + grayscale conversion
void preprocessImage(unsigned short* raw_cam_data, uint8_t* output_tensor_data) {
  // 1. Calculate the square area cropped from the center (for example, a 144x144 area is cropped from a 176x144 area).
  int crop_size = min(CAM_WIDTH, CAM_HEIGHT); // 144
  int offset_x = (CAM_WIDTH - crop_size) / 2; // (176-144)/2 = 16
  int offset_y = (CAM_HEIGHT - crop_size) / 2; // 0

  // 2. Calculate scaling step (Nearest Neighbor)
  float step_x = (float)crop_size / MODEL_WIDTH;
  float step_y = (float)crop_size / MODEL_HEIGHT;

  for (int y = 0; y < MODEL_HEIGHT; y++) {
    for (int x = 0; x < MODEL_WIDTH; x++) {
      // Mapping back to the original image coordinates
      int src_x = offset_x + (int)(x * step_x);
      int src_y = offset_y + (int)(y * step_y);
      
      if (src_x >= CAM_WIDTH) src_x = CAM_WIDTH - 1;
      if (src_y >= CAM_HEIGHT) src_y = CAM_HEIGHT - 1;

      int src_index = src_y * CAM_WIDTH + src_x;
      unsigned short pixel = raw_cam_data[src_index];

      // =======================================================
      // Byte Swap
      // =======================================================
      uint16_t swapped_pixel = (pixel >> 8) | (pixel << 8);

      uint8_t r = (swapped_pixel & 0xF800) >> 8;
      uint8_t g = (swapped_pixel & 0x07E0) >> 3;
      uint8_t b = (swapped_pixel & 0x001F) << 3;
      
      int dst_index = y * MODEL_WIDTH + x;
      // tf中使用了人眼亮度感知公式
      //output_tensor_data[dst_index] = (r + g + b) / 3;
      output_tensor_data[dst_index] = (uint8_t)(0.2989 * r + 0.5870 * g + 0.1140 * b);
    }
  }
}

void setup() {
  Serial.begin(115200);
  while (!Serial);

  Serial.println("Starting...");

  if (!Camera.begin(QCIF, RGB565, 1)) {
    Serial.println("Failed to initialize camera!");
    while (1);
  }
  Serial.println("Camera initialized.");

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
}

void loop() {
  Camera.readFrame(camera_buffer);

  preprocessImage(camera_buffer, processed_image);

#ifdef DEBUG_MODE
  // --- Send the cropped and scaled image to the Python script ---
  Serial.println("START_IMAGE");
  for (int i = 0; i < MODEL_PIXELS; i++) {
    Serial.println(processed_image[i]); 
  }
  Serial.println("END_IMAGE");
  delay(1000); 
  return;
#endif

  float input_scale = tflInputTensor->params.scale;
  int input_zero_point = tflInputTensor->params.zero_point;

  for (int i = 0; i < tflInputTensor->bytes; i++) {
    float normalized = processed_image[i] / 255.0f;
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

  float max_score = rock;
  int max_index = 0;

  if (paper > max_score) { max_score = paper; max_index = 1; }
  if (scissors > max_score) { max_score = scissors; max_index = 2; }

  Serial.print("Prediction: ");
  Serial.print(CLASSES[max_index]);
  Serial.print(" (Confidence: ");
  Serial.print(max_score * 100, 1);
  Serial.println("%)");

  delay(1000); 
}