import google.generativeai as genai
from PIL import Image
import torch
from diffusers import StableDiffusionPipeline  

API_KEY = "AIzaSyDOis7Mvfm83_ipHtQg8f82gTnQC6jC6Io"
genai.configure(api_key=API_KEY)


def extract_features(image_path,theme='abstract', model_name="gemini-1.5-flash"):

    myfile = Image.open(image_path)
    print(f"Uploaded file: {myfile}")

    model = genai.GenerativeModel(model_name)
    prompt = f"\n\nCan you tell me about the shapes present in this photo along with their color?\n\nGive me their general location within the drawing as well\n\nTheme of the prompt is {theme},try to incorporate the elements found in image in such a way so that they fit into the theme.Don't give me just descriptions try and improve upon it.Have the response be limited to 70 tokens,you may skip any unimportant information"

    result = model.generate_content(
        [myfile, prompt]
    )
    print(f"Result: {result.text}")
    return result.text

def generate_image(prompt1,theme='abstract'):
    model_id = "runwayml/stable-diffusion-v1-5"  
    pipeline = StableDiffusionPipeline.from_pretrained(model_id, torch_dtype=torch.float16).to("cuda")

    seed = torch.randint(0, 1000000, (1,)).item()
    generator = torch.Generator("cuda").manual_seed(seed)

    prompted = f"{theme}, containing {prompt1}"

    # Define the prompts
    prompt = prompted + "beautiful,4k"
    negative_prompt = "ugly, deformed, disfigured, poor details,missing features"

    image = pipeline(
        prompt=prompt,
        negative_prompt=negative_prompt,
        generator=generator,
        guidance_scale=3.5
    ).images[0]

    image.save('generated.png')
    return image

if __name__ == '__main__':
    theme = 'medievil'
    image = generate_image(extract_features("drawing.png",theme=theme),theme=theme)
    image.show()