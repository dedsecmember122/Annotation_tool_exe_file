"""
HuggingFace zero-shot / open-vocabulary detector.

Default model: google/owlvit-base-patch32

LICENSE NOTE:
  OWL-ViT (google/owlvit-base-patch32) is released under the Apache 2.0
  license and IS commercially usable. Verify the specific checkpoint's
  Model Card on HuggingFace before deploying commercially.

  Grounding DINO (IDEA-Research/grounding-dino-base) is also Apache 2.0.

To use:
  pip install transformers torch Pillow
"""
from backend.app.ml.base_detector import BaseDetector


class HFZeroShotDetector(BaseDetector):

    def __init__(self, model_name: str = "google/owlvit-base-patch32") -> None:
        self._model_name = model_name
        self._processor = None
        self._model = None

    def load(self, model_dir: str = "") -> None:
        try:
            from transformers import OwlViTForObjectDetection, OwlViTProcessor  # type: ignore
            self._processor = OwlViTProcessor.from_pretrained(self._model_name)
            self._model = OwlViTForObjectDetection.from_pretrained(self._model_name)
        except ImportError as exc:
            raise RuntimeError(
                "transformers and torch are required for HF zero-shot detection. "
                "Install them with:  pip install transformers torch"
            ) from exc

    def predict(self, image_path: str, text_queries: list[str] | None = None) -> list[dict]:
        if not text_queries:
            return []
        if self._model is None:
            self.load()

        import torch
        from PIL import Image as PILImage

        image = PILImage.open(image_path).convert("RGB")
        inputs = self._processor(text=[text_queries], images=image, return_tensors="pt")

        with torch.no_grad():
            outputs = self._model(**inputs)

        target_sizes = torch.tensor([image.size[::-1]])
        results = self._processor.post_process_object_detection(
            outputs, threshold=0.1, target_sizes=target_sizes
        )[0]

        detections: list[dict] = []
        for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
            x1, y1, x2, y2 = box.tolist()
            detections.append({
                "class": text_queries[label.item()],
                "bbox": [x1, y1, x2, y2],
                "confidence": score.item(),
            })
        return detections

    def train(self, annotated_images: list[dict], output_dir: str) -> str:
        raise NotImplementedError(
            "HFZeroShotDetector does not support training. "
            "Use the CustomModelAdapter for fine-tuning."
        )
