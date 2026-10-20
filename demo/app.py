from pathlib import Path
from typing import Dict, Tuple
import joblib
import numpy as np
from PIL import Image

import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.config import RESULTS_DIR, SAMPLE_CLASSES, VIETNAMESE_LABELS
from src.data import extract_color_texture_features, get_transforms
from src.model import create_deep_model

try:
    import torch
    import torch.nn.functional as F
except ImportError:
    torch = None

try:
    import gradio as gr
except ImportError:
    gr = None


class DiseasePredictor:
    def __init__(self):
        self.pt_ckpt_path = RESULTS_DIR / "best_model.pth"
        self.sklearn_ckpt_path = RESULTS_DIR / "color_logreg_pipeline.joblib"
        self.model_type = "none"
        self.class_names = SAMPLE_CLASSES
        self.model = None
        self.eval_transform = None
        self._load_saved_model()

    def _load_saved_model(self) -> None:
        if self.sklearn_ckpt_path.exists():
            bundle = joblib.load(self.sklearn_ckpt_path)
            self.class_names = bundle.get("class_names", SAMPLE_CLASSES)

        pt_f1 = -1.0
        if torch is not None and self.pt_ckpt_path.exists():
            ckpt = torch.load(self.pt_ckpt_path, map_location="cpu", weights_only=False)
            pt_f1 = float(ckpt.get("val_metrics", {}).get("f1_macro", -1.0))

        if torch is not None and self.pt_ckpt_path.exists() and pt_f1 >= 0.85:
            ckpt = torch.load(self.pt_ckpt_path, map_location="cpu", weights_only=False)
            arch_name = ckpt.get("arch_name", "resnet18")
            if "class_names" in ckpt:
                self.class_names = ckpt["class_names"]
            num_classes = ckpt.get("num_classes", len(self.class_names))
            self.model = create_deep_model(arch_name=arch_name, num_classes=num_classes, pretrained=False)
            self.model.load_state_dict(ckpt["state_dict"])
            self.model.eval()
            _, self.eval_transform = get_transforms(use_augmentation=False)
            self.model_type = "ResNet18 (timm)"
        elif self.sklearn_ckpt_path.exists():
            bundle = joblib.load(self.sklearn_ckpt_path)
            self.model = bundle["pipeline"]
            self.model_type = "HSV + Logistic Regression"

    def predict(self, image: Image.Image) -> Tuple[Dict[str, float], str]:
        if image is None:
            return {}, "Vui lòng tải ảnh lên để xem kết quả."

        temperature = 0.5
        if self.model_type.startswith("ResNet") and self.model is not None:
            tensor_img = self.eval_transform(image.convert("RGB")).unsqueeze(0)
            with torch.no_grad():
                logits = self.model(tensor_img)
                probs = F.softmax(logits / temperature, dim=1).squeeze(0).numpy()
        elif self.model is not None:
            temp_path = RESULTS_DIR / "_temp_demo_input.jpg"
            image.convert("RGB").save(temp_path)
            feat = extract_color_texture_features(temp_path).reshape(1, -1)
            probs = self.model.predict_proba(feat)[0]
        else:
            probs = np.ones(len(self.class_names)) / len(self.class_names)

        top_indices = np.argsort(probs)[::-1][:3]
        top_probs = probs[top_indices]
        top_probs = top_probs / np.sum(top_probs)

        top_dict = {
            VIETNAMESE_LABELS.get(self.class_names[idx], self.class_names[idx]): float(prob)
            for idx, prob in zip(top_indices, top_probs)
        }

        best_cls = self.class_names[top_indices[0]]
        best_vn_name = VIETNAMESE_LABELS.get(best_cls, best_cls)
        best_prob = float(top_probs[0]) * 100.0

        summary_md = f"### Kết quả dự đoán: **{best_vn_name}** (`{best_prob:.2f}%`)"
        return top_dict, summary_md


def build_gradio_interface():
    predictor = DiseasePredictor()

    with gr.Blocks(title="Plant Disease Classification") as demo:
        gr.Markdown("## Plant Disease Classification from Leaf Images")
        with gr.Row():
            with gr.Column(scale=5):
                img_input = gr.Image(type="pil", label="(Input)")
                btn_submit = gr.Button("Chẩn đoán", variant="primary")
            with gr.Column(scale=7):
                label_output = gr.Label(num_top_classes=3, label="(Output)")
                text_output = gr.Markdown()

        btn_submit.click(fn=predictor.predict, inputs=img_input, outputs=[label_output, text_output])
        img_input.change(fn=predictor.predict, inputs=img_input, outputs=[label_output, text_output])

    return demo


if __name__ == "__main__":
    if gr is not None:
        app = build_gradio_interface()
        app.launch(server_name="127.0.0.1", server_port=7860, share=False)
