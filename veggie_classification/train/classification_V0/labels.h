#ifndef LABELS_H
#define LABELS_H

const int NUM_CLASSES = 10;
const float CONFIDENCE_THRESHOLD = 0.3f;
const char* UNKNOWN_LABEL = "Object not known";

const char* LABELS[] = {
  "Bell pepper",
  "Broccoli",
  "Cabbage",
  "Carrot",
  "Cucumber",
  "Eggplant",
  "Garlic",
  "Onion",
  "Potato",
  "Tomato",
};

// Usage: if (max_confidence < CONFIDENCE_THRESHOLD) return UNKNOWN_LABEL;
//        else return LABELS[predicted_class];

#endif // LABELS_H
