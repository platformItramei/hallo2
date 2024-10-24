# pylint: disable=E1101
# scripts/inference.py

"""
This script contains the main inference pipeline for processing audio and image inputs to generate a video output.

The script imports necessary packages and classes, defines a neural network model,
and contains functions for processing audio embeddings and performing inference.

The main inference process is outlined in the following steps:
1. Initialize the configuration.
2. Set up runtime variables.
3. Prepare the input data for inference (source image, face mask, and face embeddings).
4. Process the audio embeddings.
5. Build and freeze the model and scheduler.
6. Run the inference loop and save the result.

Usage:
This script can be run from the command line with the following arguments:
- audio_path: Path to the audio file.
- image_path: Path to the source image.
- face_mask_path: Path to the face mask image.
- face_emb_path: Path to the face embeddings file.
- output_path: Path to save the output video.

Example:
python scripts/inference.py --audio_path audio.wav --image_path image.jpg
    --face_mask_path face_mask.png --face_emb_path face_emb.pt --output_path output.mp4
"""

import argparse
import base64
import contextlib
import math
import os
import pickle
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from diffusers import AutoencoderKL, DDIMScheduler
from omegaconf import OmegaConf
from PIL import Image
from pydub import AudioSegment
from torch import nn

# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from hallo.animate.face_animate import FaceAnimatePipeline
from hallo.datasets.audio_processor import AudioProcessor
from hallo.datasets.image_processor import ImageProcessor
from hallo.models.audio_proj import AudioProjModel
from hallo.models.face_locator import FaceLocator
from hallo.models.image_proj import ImageProjModel
from hallo.models.unet_2d_condition import UNet2DConditionModel
from hallo.models.unet_3d import UNet3DConditionModel

# from hallo.utils.util import tensor_to_video_batch

torch.backends.cuda.matmul.allow_tf32 = True
torch._inductor.config.conv_1x1_as_mm = True
torch._inductor.config.coordinate_descent_tuning = True
torch._inductor.config.epilogue_fusion = False
torch._inductor.config.coordinate_descent_check_all_directions = True


class Net(nn.Module):
    """
    The Net class combines all the necessary modules for the inference process.

    Args:
        reference_unet (UNet2DConditionModel): The UNet2DConditionModel used as a reference for inference.
        denoising_unet (UNet3DConditionModel): The UNet3DConditionModel used for denoising the input audio.
        face_locator (FaceLocator): The FaceLocator model used to locate the face in the input image.
        imageproj (nn.Module): The ImageProjector model used to project the source image onto the face.
        audioproj (nn.Module): The AudioProjector model used to project the audio embeddings onto the face.
    """

    def __init__(
        self,
        reference_unet: UNet2DConditionModel,
        denoising_unet: UNet3DConditionModel,
        face_locator: FaceLocator,
        imageproj,
        audioproj,
    ):
        super().__init__()
        self.reference_unet = reference_unet
        self.denoising_unet = denoising_unet
        self.face_locator = face_locator
        self.imageproj = imageproj
        self.audioproj = audioproj

    def forward(
        self,
    ):
        """
        empty function to override abstract function of nn Module
        """

    def get_modules(self):
        """
        Simple method to avoid too-few-public-methods pylint error
        """
        return {
            "reference_unet": self.reference_unet,
            "denoising_unet": self.denoising_unet,
            "face_locator": self.face_locator,
            "imageproj": self.imageproj,
            "audioproj": self.audioproj,
        }


def process_audio_emb(audio_emb):
    """
    Process the audio embedding to concatenate with other tensors.

    Parameters:
        audio_emb (torch.Tensor): The audio embedding tensor to process.

    Returns:
        concatenated_tensors (List[torch.Tensor]): The concatenated tensor list.
    """
    concatenated_tensors = []

    for i in range(audio_emb.shape[0]):
        vectors_to_concat = [
            audio_emb[max(min(i + j, audio_emb.shape[0] - 1), 0)] for j in range(-2, 3)
        ]
        concatenated_tensors.append(torch.stack(vectors_to_concat, dim=0))

    audio_emb = torch.stack(concatenated_tensors, dim=0)

    return audio_emb


def save_image_batch(image_tensor, save_path):
    image_tensor = (image_tensor + 1) / 2

    os.makedirs(save_path, exist_ok=True)

    for i in range(image_tensor.shape[0]):
        img_tensor = image_tensor[i]

        img_array = img_tensor.permute(1, 2, 0).cpu().numpy()

        img_array = (img_array * 255).astype(np.uint8)

        image = Image.fromarray(img_array)
        image.save(os.path.join(save_path, f"motion_frame_{i}.png"))


def cut_audio(audio_path, save_dir, length=60):
    audio = AudioSegment.from_wav(audio_path)

    segment_length = length * 1000  # pydub使用毫秒
    # segment_length = length

    # num_segments = len(audio) // segment_length + (
    #     1 if len(audio) % segment_length != 0 else 0
    # )
    num_segments = math.ceil(len(audio) / segment_length)

    os.makedirs(save_dir, exist_ok=True)

    audio_list = []

    for i in range(num_segments):
        path = f"{save_dir}/segment_{i+1}.wav"
        if os.path.exists(path):
            audio_list.append(path)
            continue

        start_time = i * segment_length
        end_time = min((i + 1) * segment_length, len(audio))
        segment = audio[start_time:end_time]
        audio_list.append(path)
        segment.export(path, format="wav")

    return audio_list


@contextlib.contextmanager
def cached_computation(cache_dir, cache_key):
    """
    A context manager that caches the result of a computation.

    Args:
        cache_dir (str): Directory to store the cache files.
        cache_key (str): Unique identifier for the cached result.

    Yields:
        tuple: (result, cache_hit, save_result_func)
            - result: The result of the computation (None if not computed yet).
            - cache_hit: Boolean indicating whether the result was loaded from cache.
            - save_result_func: Function to save the result if it wasn't cached.
    """
    os.makedirs(cache_dir, exist_ok=True)
    cache_file = os.path.join(cache_dir, f"{cache_key}.pkl")

    result = None
    cache_hit = False

    if os.path.exists(cache_file):
        with open(cache_file, "rb") as f:
            result = pickle.load(f)
        cache_hit = True
        print(f"Loaded cached result for {cache_key}")

    def save_result(data):
        with open(cache_file, "wb") as f:
            pickle.dump(data, f)
        print(f"Saved result to cache for {cache_key}")

    yield result, cache_hit, save_result

    if not cache_hit and result is not None:
        save_result(result)


class InferencePipeline:
    def __init__(
        self,
        config: str = "configs/inference/long.yaml",
        source_image: str = None,
        pose_weight: float = None,
        face_weight: float = None,
        lip_weight: float = None,
        face_expand_ratio: float = None,
        audio_ckpt_dir: str = None,
    ):
        self.config_path = config
        self.config = OmegaConf.load(config)
        self.config.source_image = self.config.source_image or source_image
        self.config.pose_weight = self.config.pose_weight or pose_weight
        self.config.face_weight = self.config.face_weight or face_weight
        self.config.lip_weight = self.config.lip_weight or lip_weight
        self.config.face_expand_ratio = (
            self.config.face_expand_ratio or face_expand_ratio
        )
        self.config.audio_ckpt_dir = self.config.audio_ckpt_dir or audio_ckpt_dir

        self.loaded = False
        self.load()

    def load(self):
        if not self.loaded:
            self.process_source_image()
            self.build_modules()
            self.loaded = True
            self.audio_processor = AudioProcessor(
                self.config.data.driving_audio.sample_rate,
                self.config.data.export_video.fps,
                self.config.wav2vec.model_path,
                self.config.wav2vec.features == "last",
                # os.path.dirname(self.config.audio_separator.model_path),
                # os.path.basename(self.config.audio_separator.model_path),
                cache_dir=os.path.join(self.save_path, "audio_preprocess"),
            )
            print("models loaded")

        # self.pipeline.reference_unet.to(memory_format=torch.channels_last)
        # self.pipeline.denoising_unet.to(memory_format=torch.channels_last)
        # self.pipeline.vae.to(memory_format=torch.channels_last)

        # self.pipeline.reference_unet = torch.compile(self.pipeline.reference_unet)
        # self.pipeline.denoising_unet = torch.compile(self.pipeline.denoising_unet)
        # self.pipeline.vae.decode = torch.compile(self.pipeline.vae.decode, mode="reduce-overhead")

    def infer(self, driving_audio: str | None = None, cut: bool = True, inference_steps: int = 40):
        driving_audio = driving_audio or self.config.driving_audio
        yield from self.inference_process(driving_audio, inference_steps, cut)

    @property
    def weight_dtype(self):
        if self.config.weight_dtype == "fp16":
            return torch.float16
        elif self.config.weight_dtype == "bf16":
            return torch.bfloat16
        else:
            return torch.float32

    @property
    def save_path(self):
        path = os.path.join(self.config.save_path, Path(self.config.source_image).stem)
        if not os.path.exists(path):
            os.makedirs(path)
        return path

    @property
    def save_seg_path(self):
        path = os.path.join(self.save_path, "seg_video")
        if not os.path.exists(path):
            os.makedirs(path)
        return path

    @property
    def motion_scale(self):
        return [
            self.config.pose_weight,
            self.config.face_weight,
            self.config.lip_weight,
        ]

    @property
    def device(self):
        return (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )

    @property
    def img_size(self):
        return (
            self.config.data.source_image.width,
            self.config.data.source_image.height,
        )

    @property
    def cache_dir(self):
        return os.path.join(self.save_path, "computation_cache")

    @property
    def clip_length(self):
        return self.config.data.n_sample_frames

    def process_source_image(self):
        with ImageProcessor(
            self.img_size, self.config.face_analysis.model_path
        ) as image_processor, cached_computation(
            self.cache_dir, "source_image_processing"
        ) as (
            cached_result,
            cache_hit,
            save_result,
        ):
            if cache_hit:
                result = cached_result
            else:
                result = image_processor.preprocess(
                    self.config.source_image,
                    self.save_path,
                    self.config.face_expand_ratio,
                )
                save_result(result)

            (
                source_image_pixels,
                source_image_face_region,
                source_image_face_emb,
                source_image_full_mask,
                source_image_face_mask,
                source_image_lip_mask,
            ) = result

            self.source_image_pixels = source_image_pixels.unsqueeze(0)
            self.source_image_face_region = source_image_face_region.unsqueeze(0)
            self.source_image_face_emb = source_image_face_emb.reshape(1, -1)
            self.source_image_face_emb = torch.tensor(source_image_face_emb)

            self.source_image_full_mask = [
                (mask.repeat(self.clip_length, 1)) for mask in source_image_full_mask
            ]
            self.source_image_face_mask = [
                (mask.repeat(self.clip_length, 1)) for mask in source_image_face_mask
            ]
            self.source_image_lip_mask = [
                (mask.repeat(self.clip_length, 1)) for mask in source_image_lip_mask
            ]

            return (
                self.source_image_pixels,
                self.source_image_face_region,
                self.source_image_face_emb,
                self.source_image_full_mask,
                self.source_image_face_mask,
                self.source_image_lip_mask,
            )

    def process_audio_emb(self, driving_audio_path: str, cut: bool = True):
        if cut:
            audio_list = cut_audio(
                driving_audio_path,
                os.path.join(
                    self.save_path, f"seg-long-{Path(driving_audio_path).stem}"
                ),
                length=1,
            )

            audio_emb_list = []
            l = 0

            audio_processor = AudioProcessor(
                self.config.data.driving_audio.sample_rate,
                self.config.data.export_video.fps,
                self.config.wav2vec.model_path,
                self.config.wav2vec.features == "last",
                # os.path.dirname(self.config.audio_separator.model_path),
                # os.path.basename(self.config.audio_separator.model_path),
                cache_dir=os.path.join(self.save_path, "audio_preprocess"),
            )

            for idx, audio_path in enumerate(audio_list):
                padding = (idx + 1) == len(audio_list)
                emb, length = audio_processor.preprocess(
                    audio_path,
                    self.clip_length,
                    padding=padding,
                    processed_length=l,
                )
                audio_emb_list.append(emb)
                l += length

            audio_emb = torch.cat(audio_emb_list)
            audio_length = l
        else:
            audio_emb, audio_length = self.audio_processor.preprocess(
                driving_audio_path, self.clip_length
            )
    
        return audio_emb, audio_length

    def build_modules(self):
        # 4. build modules
        sched_kwargs = OmegaConf.to_container(self.config.noise_scheduler_kwargs)
        if self.config.enable_zero_snr:
            sched_kwargs.update(
                rescale_betas_zero_snr=True,
                timestep_spacing="trailing",
                prediction_type="v_prediction",
            )
        val_noise_scheduler = DDIMScheduler(**sched_kwargs)
        sched_kwargs.update({"beta_schedule": "scaled_linear"})

        vae = AutoencoderKL.from_pretrained(self.config.vae.model_path)
        reference_unet = UNet2DConditionModel.from_pretrained(
            self.config.base_model_path, subfolder="unet"
        )
        denoising_unet = UNet3DConditionModel.from_pretrained_2d(
            self.config.base_model_path,
            self.config.motion_module_path,
            subfolder="unet",
            unet_additional_kwargs=OmegaConf.to_container(
                self.config.unet_additional_kwargs
            ),
            use_landmark=False,
        )
        # denoising_unet.set_attn_processor()

        face_locator = FaceLocator(conditioning_embedding_channels=320)
        image_proj = ImageProjModel(
            cross_attention_dim=denoising_unet.config.cross_attention_dim,
            clip_embeddings_dim=512,
            clip_extra_context_tokens=4,
        )

        audio_proj = AudioProjModel(
            seq_len=5,
            blocks=12,  # use 12 layers' hidden states of wav2vec
            channels=768,  # audio embedding channel
            intermediate_dim=512,
            output_dim=768,
            context_tokens=32,
        ).to(device=self.device, dtype=self.weight_dtype)

        audio_ckpt_dir = self.config.audio_ckpt_dir

        # Freeze
        vae.requires_grad_(False)
        image_proj.requires_grad_(False)
        reference_unet.requires_grad_(False)
        denoising_unet.requires_grad_(False)
        face_locator.requires_grad_(False)
        audio_proj.requires_grad_(False)

        reference_unet.enable_gradient_checkpointing()
        denoising_unet.enable_gradient_checkpointing()

        net = Net(
            reference_unet,
            denoising_unet,
            face_locator,
            image_proj,
            audio_proj,
        )

        m, u = net.load_state_dict(
            torch.load(
                os.path.join(audio_ckpt_dir, "net.pth"),
                map_location="cpu", weights_only=True,
            ),
            
        )
        assert len(m) == 0 and len(u) == 0, "Fail to load correct checkpoint."
        print("loaded weight from ", os.path.join(audio_ckpt_dir, "net.pth"))
        pipeline = FaceAnimatePipeline(
            vae=vae,
            reference_unet=net.reference_unet,
            denoising_unet=net.denoising_unet,
            face_locator=net.face_locator,
            scheduler=val_noise_scheduler,
            image_proj=net.imageproj,
        )
        pipeline.to(device=self.device, dtype=self.weight_dtype)
        self.pipeline = pipeline
        self.net = net

    def motion_zeros(self):
        motion_zeros = self.source_image_pixels.repeat(
            self.config.data.n_motion_frames, 1, 1, 1
        )
        motions = motion_zeros.to(
            dtype=self.source_image_pixels.dtype, device=self.source_image_pixels.device
        )
        return motions

    def motion_frames(self, tensor_result):
        motion_frames = tensor_result[-1][0]
        motion_frames = motion_frames.permute(1, 0, 2, 3)
        motion_frames = motion_frames[0 - self.config.data.n_motion_frames :]
        motion_frames = motion_frames * 2.0 - 1.0
        motions = motion_frames.to(
            dtype=self.source_image_pixels.dtype, device=self.source_image_pixels.device
        )
        return motions

    def generate_mask(self, b, f, c, h, w):
        rand_mask = torch.rand(h, w)
        mask = rand_mask > self.config.mask_rate
        mask = mask.unsqueeze(0).unsqueeze(0).unsqueeze(0)
        mask = mask.expand(b, f, c, h, w)

        face_mask = self.source_image_face_region.repeat(f, 1, 1, 1).unsqueeze(0)
        assert face_mask.shape == mask.shape
        mask = mask | face_mask.bool()
        return mask

    # def save_tensor_result(
    #     self, start, t, audio_length, driving_audio_path, tensor_result
    # ):
    #     last_motion_frame = [tensor_result[-1]]

    #     if start != 0:
    #         tensor_result = torch.cat(tensor_result[1:], dim=2)
    #     else:
    #         tensor_result = torch.cat(tensor_result, dim=2)

    #     tensor_result = tensor_result.squeeze(0)
    #     f = tensor_result.shape[1]
    #     length = min(f, audio_length)
    #     tensor_result = tensor_result[:, :length]

    #     name = Path(self.save_path).name
    #     output_file = os.path.join(self.save_seg_path, f"{name}-{t+1:06}.mp4")

    #     tensor_to_video_batch(tensor_result, output_file, start, driving_audio_path)
    #     del tensor_result

    #     return last_motion_frame, length

    @torch.no_grad()
    def inference_process(
        self,
        driving_audio_path: str,
        inference_steps: int = 40,
        cut: bool = True,
    ):
        if not self.loaded:
            self.load()

        audio_emb, audio_length = self.process_audio_emb(driving_audio_path, cut)
        audio_emb = process_audio_emb(audio_emb)

        times = audio_emb.shape[0] // self.clip_length
        tensor_result = []
        generator = torch.manual_seed(42)
        start = 0
        start_time = time.time()

        for t in range(times):
            print(f"[{t+1}/{times}] {time.time() - start_time:.2f}s")
            start_time = time.time()

            if len(tensor_result) == 0:
                motions = self.motion_zeros()
            else:
                motions = self.motion_frames(tensor_result)

            pixel_values_ref_img = torch.cat(
                [self.source_image_pixels, motions], dim=0
            )  # concat the ref image and the motion frames

            pixel_values_ref_img = pixel_values_ref_img.unsqueeze(0)
            pixel_motion_values = pixel_values_ref_img[:, 1:]

            if self.config.use_mask:
                b, f, c, h, w = pixel_motion_values.shape
                mask = self.generate_mask(b, f, c, h, w)
                pixel_motion_values = pixel_motion_values * mask
                pixel_values_ref_img[:, 1:] = pixel_motion_values

            assert pixel_motion_values.shape[0] == 1

            audio_tensor = (
                audio_emb[
                    t * self.clip_length : min(
                        (t + 1) * self.clip_length, audio_emb.shape[0]
                    )
                ]
                .unsqueeze(0)
                .to(device=self.net.audioproj.device, dtype=self.net.audioproj.dtype)
            )
            audio_tensor = self.net.audioproj(audio_tensor)

            pipeline_output = self.pipeline(
                ref_image=pixel_values_ref_img,
                audio_tensor=audio_tensor,
                face_emb=self.source_image_face_emb,
                face_mask=self.source_image_face_region,
                pixel_values_full_mask=self.source_image_full_mask,
                pixel_values_face_mask=self.source_image_face_mask,
                pixel_values_lip_mask=self.source_image_lip_mask,
                width=self.img_size[0],
                height=self.img_size[1],
                video_length=self.clip_length,
                num_inference_steps=inference_steps,
                guidance_scale=self.config.cfg_scale,
                generator=generator,
                motion_scale=self.motion_scale,
                return_dict=False,
            )
            
            tensor_result.append(pipeline_output)
            
            last_motion_frame = [tensor_result[-1]]
            if start != 0:
                tensor_result = torch.cat(tensor_result[1:], dim=2)
            else:
                tensor_result = torch.cat(tensor_result, dim=2)

            tensor_result = tensor_result.squeeze(0)
            f = tensor_result.shape[1]
            length = min(f, audio_length)
            tensor_result = tensor_result[:, :length]
            tensor = tensor_result.permute(1, 2, 3, 0).cpu(
            ).numpy()  # convert to [f, h, w, c]
            tensor = np.clip(tensor * 255, 0, 255).astype(
                np.uint8
            )
            for i, frame in enumerate(tensor):
                x = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)  # Convert RGB to BGR for OpenCV
                _, data_buffer = cv2.imencode('.jpg', x, [cv2.IMWRITE_JPEG_QUALITY, 85])
                b64 = base64.b64encode(data_buffer).decode('utf-8')
                yield b64, i, length
            
            tensor_result = last_motion_frame
            audio_length -= length
            start += length

        #     tensor_result.append(pipeline_output)

        #     if (t + 1) % batch_size == 0 or (t + 1) == times:
        #         start_io_time = time.time()
        #         tensor_result, length = self.save_tensor_result(
        #             start, t, audio_length, driving_audio_path, tensor_result
        #         )
        #         audio_length -= length
        #         start += length
        #         print(f"[{t+1}/{times}] IO time: {time.time() - start_io_time:.2f}s")

        # return self.save_seg_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("-c", "--config", default="configs/inference/long.yaml")
    parser.add_argument("--source_image", type=str, required=False, help="source image")
    parser.add_argument(
        "--driving_audio", type=str, required=False, help="driving audio"
    )
    parser.add_argument(
        "--pose_weight", type=float, help="weight of pose", required=False
    )
    parser.add_argument(
        "--face_weight", type=float, help="weight of face", required=False
    )
    parser.add_argument(
        "--lip_weight", type=float, help="weight of lip", required=False
    )
    parser.add_argument(
        "--face_expand_ratio", type=float, help="face region", required=False
    )
    parser.add_argument(
        "--audio_ckpt_dir",
        "--checkpoint",
        type=str,
        help="specific checkpoint dir",
        required=False,
    )

    command_line_args = parser.parse_args()
    pipeline = InferencePipeline(
        command_line_args.config,
        command_line_args.source_image,
        command_line_args.pose_weight,
        command_line_args.face_weight,
        command_line_args.lip_weight,
        command_line_args.face_expand_ratio,
        command_line_args.audio_ckpt_dir,
    )
    pipeline.infer(
        command_line_args.driving_audio,
    )
