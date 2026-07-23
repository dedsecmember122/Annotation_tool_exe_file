"""
Abstract base class every model adapter must implement.

Usage
-----
Implement this interface in custom_model_adapter.py (for your own model)
or hf_zero_shot.py (for HuggingFace zero-shot detectors).

The contract:
  load()    — load weights/config from disk
  predict() — run inference on a single image, return list of detections
  train()   — fine-tune on a labeled dataset, return path to new checkpoint
"""
from abc import ABC, abstractmethod


class BaseDetector(ABC):

    @abstractmethod
    def load(self, model_dir: str) -> None:
        """
        Load model weights and configuration from *model_dir*.
        Called once before predict() or train().
        """
        ...

    @abstractmethod
    def predict(self, image_path: str) -> list[dict]:
        """
        Run inference on the image at *image_path*.

        Returns
        -------
        list of dicts, each with:
          {
            "class":      str,            # class name
            "bbox":       [x1, y1, x2, y2],  # pixel coords, absolute
            "confidence": float,          # 0.0 – 1.0
          }
        """
        ...

    @abstractmethod
    def train(self, annotated_images: list[dict], output_dir: str) -> str:
        """
        Fine-tune/train the model on *annotated_images*.

        Parameters
        ----------
        annotated_images:
          [
            {
              "image_path": str,
              "annotations": [
                {"class": str, "bbox": [x1,y1,x2,y2]},
                ...
              ]
            },
            ...
          ]
        output_dir:
          Destination for the new checkpoint.

        Returns
        -------
        Path to the new checkpoint directory/file.
        """
        ...
