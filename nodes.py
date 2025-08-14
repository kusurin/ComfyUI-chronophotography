import cv2
import numpy as np
import torch

class CreateChronophotography:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "frames": ("IMAGE",),
                "masks": ("MASK",),
                "mode": (["arange", "list"], {"default": "arange"}),
            },
            "optional": {
                "start_frame": ("INT", {"default": 0}),
                "step_frame": ("INT", {"default": 30}),
                "stop_frame": ("INT", {"default": 100}),
                "render_frames": ("STRING", {"default": "0,30,60,90", "multiline": False}),
                "gaussian_blur_size": ("INT", {"default": 21, "min": 1, "max": 101, "step": 2}),
                "morphology_kernel_size": ("INT", {"default": 5, "min": 1, "max": 15, "step": 2}),
            }
        }
    
    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("chronophotography",)
    FUNCTION = "create_chronophotography"
    CATEGORY = "chronophotography"

    def process_mask(self, mask, kernel_size=5):
        """处理单个遮罩：形态学操作 + 选择最大连通区域"""
        if isinstance(mask, torch.Tensor):
            mask_np = (mask.cpu().numpy() * 255).astype(np.uint8)
        else:
            mask_np = (mask * 255).astype(np.uint8)
        
        if len(mask_np.shape) == 3:
            mask_np = mask_np.squeeze()
        
        kernel = np.ones((kernel_size, kernel_size), np.uint8)
        
        mask_opened = cv2.morphologyEx(mask_np, cv2.MORPH_OPEN, kernel)
        
        mask_closed = cv2.morphologyEx(mask_opened, cv2.MORPH_CLOSE, kernel)
        
        mask_dilated = cv2.dilate(mask_closed, kernel, iterations=1)
        
        # 取最大的一块
        contours, _ = cv2.findContours(mask_dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            mask_largest = np.zeros_like(mask_dilated)
            cv2.fillPoly(mask_largest, [largest_contour], 255)
            mask_dilated = mask_largest
        
        return mask_dilated

    def create_chronophotography(self, frames, masks, mode, start_frame=0, step_frame=1, stop_frame=10, 
                               render_frames="0,30,60,90", gaussian_blur_size=21, morphology_kernel_size=5):
        if mode == "arange":
            frame_indices = list(range(start_frame, min(stop_frame, len(frames)), step_frame))
        elif mode == "list":
            try:
                frame_indices = [int(x.strip()) for x in render_frames.split(',')]
                frame_indices = [i for i in frame_indices if 0 <= i < len(frames)]
            except:
                frame_indices = [0]
        
        if not frame_indices:
            frame_indices = [0]
        
        background_idx = frame_indices[-1] if frame_indices else 0
        background_frame = frames[background_idx]
        if isinstance(background_frame, torch.Tensor):
            result = background_frame.cpu().numpy()
        else:
            result = background_frame.copy()
        if len(result.shape) == 4:  # batch dimension
            result = result[0]
        if result.shape[-1] != 3:
            result = result[..., :3] if result.shape[-1] > 3 else np.repeat(result[..., None], 3, axis=-1)
        reversed_indices = [idx for idx in reversed(frame_indices[:-1])]
        
        for i, frame_idx in enumerate(reversed_indices):
            if frame_idx >= len(frames) or frame_idx >= len(masks):
                continue
                
            frame = frames[frame_idx]
            mask = masks[frame_idx]
            if isinstance(frame, torch.Tensor):
                frame_np = frame.cpu().numpy()
            else:
                frame_np = frame.copy()
            
            if len(frame_np.shape) == 4:
                frame_np = frame_np[0]
            if frame_np.shape[-1] != 3:
                frame_np = frame_np[..., :3] if frame_np.shape[-1] > 3 else np.repeat(frame_np[..., None], 3, axis=-1)
            processed_mask = self.process_mask(
                mask, 
                kernel_size=morphology_kernel_size
            )
            mask_blurred = cv2.GaussianBlur(processed_mask, (gaussian_blur_size, gaussian_blur_size), 0)
            alpha = mask_blurred.astype(np.float32) / 255.0
            alpha_3ch = np.stack([alpha, alpha, alpha], axis=-1)
            if frame_np.shape != result.shape:
                h, w = result.shape[:2]
                frame_np = cv2.resize(frame_np, (w, h))
                alpha_3ch = cv2.resize(alpha_3ch, (w, h))
            result = alpha_3ch * frame_np + (1 - alpha_3ch) * result
        result = np.clip(result, 0, 1).astype(np.float32)
        result_tensor = torch.from_numpy(result).unsqueeze(0)  # 添加 batch 维度
        
        return (result_tensor,)


NODE_CLASS_MAPPINGS = {
    "CreateChronophotography": CreateChronophotography
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "CreateChronophotography": "Create Chronophotography"
}