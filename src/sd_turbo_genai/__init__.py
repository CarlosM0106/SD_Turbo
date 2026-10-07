 import torch #Ejecutar operaciones matemáticas del modelo de IA
 from diffusers import AutoPipelineForText2Image #Diffusers es una libreria especializada en modelos generativos

 print("Cargando el modelo ...")

 modelo = AutoPipelineForText2Image.from_pretrained(
     "stabilityai/sd-turbo",
     torch_dtype=torch.float32 
 )

 modelo = modelo.to("cpu")

 prompt = input("Escribe el prompt de la imagen que quieras crear: ")
 negative_prompt = "ugly, tiling, poorly drawn hands, poorly drawn feet, poorly drawn face, out of frame, extra limbs, disfigured, deformed, body out of frame, bad anatomy, watermark, signature, cut off, low contrast, underexposed, overexposed, bad art, beginner, amateur, distorted face, blurry, draft"

 print("Generando imagen ...")

 imagen = modelo(
     prompt=prompt,
     negative_prompt=negative_prompt,
     num_inference_steps=10,
     guidance_scale=2.0,
     height=1024,
     width=1024 
 ).images[0]

 imagen.save("imagen.png")

 print("Imagen guardada.")




