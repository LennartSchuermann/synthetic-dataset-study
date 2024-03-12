import torch
import time
import cv2
import numpy as np

from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
from random import randrange

folders = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]

model_path = "redstonehero/epicrealism_pureevolutionv5" # or replace with local safetensors

# prompts
emotions = ["angry", "disgusted", "frightened", "happy", "neutral", "sad", "surprised"]
emotion_helper = [
    "screaming, ranging, yelling",
    "sickend, nauseous feeling, grossed out look",
    "in shock, fear, horrified look",
    "smiling, laughing, joyful look",
    "disinterested, inactive look",
    "crying, down, unhappy look",
    "astonished, flabbergast look",
]

sexes = ["male", "female"]
ages = ["baby", "child", "teenage", "young adult", "adult", "middle aged", "eldery", "old"]
ethnicities = ["white", "european", "caucasian", "asian", "black", "mixed", "arab", "hispanic", "african", "indian"]

negative_prompt = "text, watermark, cutout, sketch, worst quality, glitches, mutations, extra limbs, missing limbs, missing fingers, body, cgi, 3d render, unreal"

# setup
device = "cuda" if torch.cuda.is_available() else "cpu"
generator = torch.Generator(device=device)

inference_steps = 25
batch_size = 8
scale = 7
width = 256
height = 256

def save_img(img, emotion_index, index, training : bool):
    folder_path = "dataset_synthetic/"
    if(training): folder_path += "train/" 
    else: folder_path += "validation/" # not used
    
    folder_path += folders[emotion_index] + "/"
    
    cv2.imwrite(folder_path + str(index) + ".png", img)
    

def generate_prompt():
    emotion_index = randrange(len(emotions))
    emotion = emotions[emotion_index]
    
    age_index = randrange(len(ages))
    age = ages[age_index]
    
    ethnicity_index = randrange(len(ethnicities))
    ethnicity = ethnicities[ethnicity_index]
    
    sex_index = randrange(len(sexes))
    sex = sexes[sex_index]
    
    prompt = emotion + " " + age + " " + ethnicity + " " + sex + ", portrait of the whole face, round realistic eyes, photo, ultra-realistic, 8k, amazing details, amazing skin texture, natural mouth, natural hair, " + emotion_helper[emotion_index]
    
    print("Prompt: " + prompt)
    return emotion_index, prompt

if __name__ == "__main__":
    try: 
        # init
        pipeline = StableDiffusionPipeline.from_single_file(model_path)
        pipeline.to(device)
        pipeline.scheduler = DPMSolverMultistepScheduler.from_config(pipeline.scheduler.config)
        
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        print("Starting Diffusion...")
        start_time = time.time()
        
        # generation process
        generations = 12
        i = 0
        j = 3968
        while True: # i < generations:
            # generate prompt
            emotion_index, prompt = generate_prompt()
            
            # generate images
            images = pipeline(prompt=prompt, negative_prompt=negative_prompt, width=width, height=height, num_inference_steps=inference_steps,
                    guidance_scale=scale, num_images_per_prompt=batch_size, generator=generator).images
            
            # save images if face gets detected
            for image in images:
                # PIL -> OpenCV
                gray_frame = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
                
                if len(list(faces)) > 0:
                    x, y, w, h = list(faces)[0]
                
                    # Crop Face
                    ROI = gray_frame[y:y+h, x:x+w]
                    ROI = cv2.resize(ROI, (64, 64))
                
                    #cv2.imwrite("dataset_synthetic/test_images/" + str(j) + "_" + str(emotion_index) + ".png", ROI)
                    save_img(ROI, emotion_index, j, True)
                    j += 1

            i += 1
            print(str(i) + "/" + str(generations))
        
    except KeyboardInterrupt:
        # Process:
        # First Pass: 100.16 min -> 1323 imgs -> 35 failures
        # Second Pass: 30.36 min -> 435 imgs -> 18 failures
        # Third Pass: 40.23 min -> 523 imgs -> 18 failures
        # Fourth Pass: 152.49 min -> 1686 imgs -> 46 failures
        # Fifth Pass: 85.44 min -> 1283 imgs -> 36 failures
        
        end_time = time.time()
        print(f"Finished in: {((end_time-start_time) / 60):.2f} min for generating {j} images.")