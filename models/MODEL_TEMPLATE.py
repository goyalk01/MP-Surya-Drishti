"""
Model Template — How to Add a New Segmentation Model to the Framework
======================================================================

To add a new model (e.g., DeepLabV3+, Mask2Former, U-Net), follow
these 4 steps:

STEP 1: Create a new file in ``models/`` (e.g., ``models/deeplabv3plus_model.py``).
STEP 2: Extend ``BaseSegmentationModel`` and decorate with ``@register_model``.
STEP 3: Implement all abstract methods (see example below).
STEP 4: Import your module in ``models/registry.py::ensure_models_registered()``.

That's it. The entire pipeline (train, infer, evaluate, CLI) will
automatically work with your new model.

---

EXAMPLE IMPLEMENTATION (DeepLabV3+ with torchvision):

    @register_model("deeplabv3plus")
    class DeepLabV3PlusModel(BaseSegmentationModel):

        @property
        def model_type(self) -> str:
            return "deeplabv3plus"

        def __init__(self, backbone="resnet101", num_labels=2, **kwargs):
            super().__init__(backbone=backbone, num_labels=num_labels, **kwargs)
            from torchvision.models.segmentation import deeplabv3_resnet101
            self.model = deeplabv3_resnet101(pretrained=True)
            self.model.classifier[-1] = nn.Conv2d(256, num_labels, 1)

        def forward(self, pixel_values, labels=None):
            output = self.model(pixel_values)
            logits = output["out"]  # (B, C, H, W) — already full resolution
            return {"upsampled_logits": logits}

        def _get_state_dict(self):
            return self.model.state_dict()

        def _load_state_dict_from_checkpoint(self, state_dict):
            self.model.load_state_dict(state_dict)

        @classmethod
        def from_checkpoint(cls, checkpoint_path, device=None):
            checkpoint = torch.load(checkpoint_path, map_location=device or "cpu")
            model = cls(backbone=checkpoint["backbone"], num_labels=checkpoint["num_labels"])
            model._load_state_dict_from_checkpoint(checkpoint["model_state_dict"])
            if device: model = model.to(device)
            return model

---

The key contract your model must satisfy:
    1. forward() → must return {"upsampled_logits": Tensor(B, C, H, W)}
    2. predict() → inherited from base class (uses upsampled_logits + softmax + threshold)
    3. save_checkpoint() → inherited from base class (uses _get_state_dict())
    4. load_checkpoint() → inherited from base class (uses _load_state_dict_from_checkpoint())
    5. from_checkpoint() → you implement this to reconstruct the model from a checkpoint file

---

Config change to use your new model:
    # configs/model_config.yaml
    model:
      model_type: "deeplabv3plus"   # <-- Just change this
      backbone: "resnet101"
      num_labels: 2
      ...

Everything else (training, inference, evaluation, postprocessing, visualization)
works without any code changes.
"""
