150 epochs completed in 0.604 hours.
Optimizer stripped from /home/awesome/training-module/runs/detect/runs/pet_feeder/yolo26n_v1/weights/last.pt, 5.4MB
Optimizer stripped from /home/awesome/training-module/runs/detect/runs/pet_feeder/yolo26n_v1/weights/best.pt, 5.4MB

Validating /home/awesome/training-module/runs/detect/runs/pet_feeder/yolo26n_v1/weights/best.pt...
Ultralytics 8.4.19 🚀 Python-3.14.3 torch-2.10.0+cu128 CUDA:0 (NVIDIA GeForce RTX 5060 Ti, 15850MiB)
YOLO26n summary (fused): 122 layers, 2,375,421 parameters, 0 gradients, 5.2 GFLOPs
Class Images Instances Box(P R mAP50 mAP50-95): 100% ━━━━━━━━━━━━ 5/5 10.4it/s 0.5s
all 214 183 0.993 0.991 0.992 0.865
bowl_empty 44 44 0.995 1 0.995 0.847
bowl_full 54 54 0.995 1 0.995 0.903
yuri-dog 85 85 0.988 0.974 0.985 0.846
Speed: 0.1ms preprocess, 0.7ms inference, 0.0ms loss, 0.0ms postprocess per image
Results saved to /home/awesome/training-module/runs/detect/runs/pet_feeder/yolo26n_v1

(training-module) awesome@awesome-server:~/training-module$ python3 -c "from train import export_to_onnx; export_to_onnx()"
Ultralytics 8.4.19 🚀 Python-3.14.3 torch-2.10.0+cu128 CPU (AMD Ryzen 5 7600 6-Core Processor)
YOLO26n summary (fused): 122 layers, 2,375,421 parameters, 0 gradients, 5.2 GFLOPs

PyTorch: starting from '/home/awesome/training-module/runs/detect/runs/pet_feeder/yolo26n_v1/weights/best.pt' with input shape (1, 3, 640, 640) BCHW and output shape(s) (1, 300, 6) (5.1 MB)

ONNX: starting export with onnx 1.20.1 opset 19...
/home/awesome/training-module/.venv/lib/python3.14/site-packages/torch/onnx/\_internal/torchscript_exporter/symbolic_opset11.py:954: UserWarning: Exporting aten::index operator of advanced indexing in opset 19 is achieved by combination of multiple ONNX operators, including Reshape, Transpose, Concat, and Gather. If indices include negative values, the exported graph will produce incorrect results.
return opset9.index(g, self, index)
ONNX: slimming with onnxslim 0.1.86...
ONNX: export success ✅ 0.6s, saved as '/home/awesome/training-module/runs/detect/runs/pet_feeder/yolo26n_v1/weights/best.onnx' (9.4 MB)

Export complete (0.7s)
Results saved to /home/awesome/training-module/runs/detect/runs/pet_feeder/yolo26n_v1/weights
Predict: yolo predict task=detect model=/home/awesome/training-module/runs/detect/runs/pet_feeder/yolo26n_v1/weights/best.onnx imgsz=640
Validate: yolo val task=detect model=/home/awesome/training-module/runs/detect/runs/pet_feeder/yolo26n_v1/weights/best.onnx imgsz=640 data=/home/awesome/training-module/Dog-face-detection-10/data.yaml  
Visualize: https://netron.app
