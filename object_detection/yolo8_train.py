from ultralytics import YOLO

model = YOLO('yolo8x.pt') 

results = model.train(data='config.yaml', 
                      epochs=300, 
                      batch=18, 
                      imgsz=1024, 
                      device=[0, 1, 2], 
                      name="yolo_NAS_L_V0", 
                      optimizer='Adam', 
                      amp=False, 
                      dropout=0.3,
                      lr0=0.001,
                      lrf=1e-12,
                      augment=True,
                      save_period=1)
