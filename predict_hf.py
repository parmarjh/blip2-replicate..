# Prediction interface for Cog ⚙️
# https://github.com/replicate/cog/blob/main/docs/python.md
import os

cache = "/src/blip2-flan-t5-xl-coco/"
os.environ["TORCH_HOME"] = "/src/blip2-flan-t5-xl-coco/"
os.environ["HF_HOME"] = "/src/blip2-flan-t5-xl-coco/"
os.environ["HUGGINGFACE_HUB_CACHE"] = "/src/blip2-flan-t5-xl-coco/"
if not os.path.exists(cache):
    os.makedirs(cache)

import torch
from cog import BasePredictor, Input, Path
from PIL import Image
from transformers import Blip2Processor, Blip2ForConditionalGeneration

class Predictor(BasePredictor):
    def setup(self):
        """Load the model into memory to make running multiple predictions efficient"""

        repository = os.path.join(".", "blip2-flan-t5-xl-coco")
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.processor = Blip2Processor.from_pretrained(repository)
        self.model = Blip2ForConditionalGeneration.from_pretrained(repository, torch_dtype=torch.float16)
        self.model.to(self.device)

    def predict(
        self,
        image: Path = Input(description="Input image to query or caption"),
        caption: bool = Input(
            description="Select if you want to generate image captions instead of asking questions",
            default=False,
        ),
        question: str = Input(
            description="Question to ask about this image. Leave blank for captioning",
            default="What is this a picture of?",
        ),
        context: str = Input(
            description="Optional - previous questions and answers to be used as context for answering current question",
            default=None,
        )
    ) -> str:
        """Run a single prediction on the model"""
        raw_image = Image.open(image).convert("RGB")

        if caption or question == "":
            inputs = self.processor(
                images=raw_image, return_tensors="pt"
            ).to(self.device, torch.float16)

        elif question:
            q = f"Question: {question} Answer:"
            inputs = self.processor(
                images=raw_image, return_tensors="pt", text=q
            ).to(self.device, torch.float16)

        elif context:
            q = " ".join([context, q])
            inputs = self.processor(
                images=raw_image, return_tensors="pt", text=q
            ).to(self.device, torch.float16)

        generated_ids = self.model.generate(**inputs)
        response = self.processor.batch_decode(
            generated_ids, skip_special_tokens=True
        )[0].strip()

        return response
