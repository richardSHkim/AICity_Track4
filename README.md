# AICity_Track4

- data yaml file must contain `val_json` for F1 metric evaluation.

### Evaluation
Evaluate model with ground-truth.
```bash
bash scripts/eval.sh
```

### Evaluation
Convert model to tensorrt (fp16, int8)
```bash
bash scripts/convert.sh
```

### Inference
Inference script for submission. Calculate FPS and generates prediction json file.
```bash
bash scripts/convert.sh
```