from backbones import get_model
import torch
import cv2
import numpy as np
from PIL import Image
from face_alignment import align
from backbones import get_model
from losses import get_loss
from utils.get_configs import Config
import json
from PIL import Image
from torchvision import transforms
from utils.preprocessing import rezise_image
import pyrealsense2 as rs
import time



def inference(model, model_cls, source):
    model.eval()
    model_cls.eval()
    transform = transforms.Compose([
        transforms.Resize((112,112)),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.unsqueeze(0)) , # Add a batch dimension
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    with open('config/classes.json', 'r') as f:
        classes = json.load(f)
    print(f"Classes : {classes}")
    
    if source == 'realsense':
        pipeline = rs.pipeline()

    # Create a config object
        config = rs.config()

        # Configure the pipeline to stream color and depth
        # Start streaming
        config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)

        # Start streaming
        pipeline.start(config)

        try:
            fps_values = []
            while True:
                # Wait for a coherent pair of frames
                frames = pipeline.wait_for_frames()
                color_frame = frames.get_color_frame()

                # Convert images to numpy arrays
                color_image = np.asanyarray(color_frame.get_data())

                # Apply a color map to depth image
                start_comp = time.time()
                with torch.no_grad():
                    face = align.get_aligned_face(Image.fromarray(color_image))
                
                # if face is not None:
                #     print("Detected")
                #     x = transform(face)

                #     with torch.no_grad():
                #         start_time = time.time()
                #         emb = model(x)
                #         end_time = time.time()
                #         print("Embedding Time" , end_time - start_time)
                #         start_time = time.time()

                #         y_pred  = model_cls(emb)
                #         probs = torch.softmax(y_pred, dim=1)
                #         pred = torch.argmax(probs, dim=1)
                #         name = classes[str(int(pred))]
                #         end_time = time.time()
                #         print("Classification Time", end_time - start_time)
                #         print(f"Name : {name} , Prob : {probs} , Probs : {max(probs)}")
                time.sleep(0.03)
                end_computing = time.time()
                elapsed_time = end_computing - start_comp
                fps = 1.0 / elapsed_time

                # Store FPS values for averaging later if desired
                fps_values.append(fps)

                # Calculate the average FPS
                avg_fps = sum(fps_values) / len(fps_values) if fps_values else 0

                # Display FPS on the frame
                cv2.putText(color_image, f"FPS: {avg_fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.imshow('Image with Bounding Boxes', color_image)
                # print(f"Name : {name} , Prob : {probs} , Probs : {max(probs)}")
                if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                # Stack both images horizontally

                # Wait for the key press

                
        finally: 
            # Stop streaming
            pipeline.stop()
            cv2.destroyAllWindows()

    
    else:
        cap = cv2.VideoCapture(source)
        count = 0
        # Load Classes forom JSON

        while cap.isOpened():
            ret, frame = cap.read()
            if ret:
                # Convert the frame to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_rgb = rezise_image(frame_rgb , "cv2")
                # Make predictions on the frame
                with torch.no_grad():
                    face = align.get_aligned_face(Image.fromarray(frame_rgb))

                if face is not None:
                    x = transform(face)

                    with torch.no_grad():
                        start_time = time.time()
                        emb = model(x)
                        end_time = time.time()
                        print("Embedding Time" , end_time - start_time)
                        start_time = time.time()

                        y_pred  = model_cls(emb)
                        probs = torch.softmax(y_pred, dim=1)
                        pred = torch.argmax(probs, dim=1)
                        name = classes[str(int(pred))]
                        end_time = time.time()

                        print("Classification Time", end_time - start_time)

                    cv2.imshow('Image with Bounding Boxes', cv2.cvtColor(np.array(face), cv2.COLOR_RGB2BGR))
                    print(f"Name : {name} , Prob : {probs} , Probs : {max(probs)}")
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
            else:
                break
        cap.release()
        cv2.destroyAllWindows

            

if __name__ == '__main__':
    cfg = Config()
    parser = Config.parse_args()
    Config.update_config_from_args(cfg, parser)

    model = get_model("edgeface_xs_gamma_06")
    model.load_state_dict(torch.load("experiments/experiment_3/checkpoints/last_model.pt"))

    model_cls = get_loss(cfg)
    model_cls.load_state_dict(torch.load("experiments/experiment_3/checkpoints/last_criterion.pt"))
    ## Concate model
    concate_model = torch.nn.Sequential(model , model_cls).to(cfg.DEVICE)



    output_onnx = 'ArchFace.onnx'
    print("==> Exporting model to ONNX format at '{}'".format(output_onnx))

    input_names = ["input1"]
    output_names = ["output1"]
    inputs = torch.randn(1 , 512).to(cfg.DEVICE)

    torch_out = torch.onnx.export(model_cls, inputs, output_onnx, export_params=True, verbose=False,
                                   input_names=input_names, output_names=output_names, dynamic_axes={'input0': {0: 'batch_size'}})
    
    ## Optional Quantize model
    # torch.save(concate_model.state_dict(), "experiments/experiment_3/checkpoints/last_concate_model.pt")
    # inference(model, model_cls, "realsense")