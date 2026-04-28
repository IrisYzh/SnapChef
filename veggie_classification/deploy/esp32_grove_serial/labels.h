#ifndef LABELS_H
#define LABELS_H

// Must match the class order the Grove Vision AI V2 model was trained with.
// For veggie_mnv2_sensecraft_int8_vela.tflite the order comes from
// classification_V3: alphabetical, Cucumber dropped.
const int NUM_CLASSES = 9;
const float CONFIDENCE_THRESHOLD = 0.6f;
const char* UNKNOWN_LABEL = "Object not known";

const char* LABELS[] = {
  "Bellpepper",
  "Broccoli",
  "Cabbage",
  "Carrot",
  "Eggplant",
  "Garlic",
  "Onion",
  "Potato",
  "Tomato",
};

#endif // LABELS_H
