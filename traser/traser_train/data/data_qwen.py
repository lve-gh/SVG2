import copy
import json
import os
import random
import re
import time
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Dict, List, Optional, Set

import numpy as np
import torch
import torch.nn.functional as F
import transformers
from decord import VideoReader
from pycocotools import mask as maskUtils
from torch.utils.data import Dataset

from . import data_list

IGNORE_INDEX = -100

# Object id representing the camera/observer in relationship triplets. It has no
# object entry or mask and is exempt from mask filtering and ID renumbering.
CAMERA_ID = -1

OBJ_ID_RE = re.compile(r"object[_ ](\d+)")

# Videos that are known to be broken or to produce sequences too long to train on.
# Matched by file name, so the list is independent of where the datasets are stored.
SKIP_VIDEO_FILENAMES = {
    # PE-Video
    "535362894.mp4", "623965466.mp4", "211998554.mp4", "623893376.mp4",
    "712368492.mp4", "406668756.mp4", "202108108.mp4", "347699158.mp4",
    "387319716.mp4", "154992052.mp4", "673630386.mp4", "344982336.mp4",
    "442536494.mp4", "438465410.mp4",
    # SA-V
    "sav_046096.mp4", "sav_006252.mp4", "sav_006195.mp4", "sav_005441.mp4",
    "sav_005458.mp4", "sav_016753.mp4", "sav_030503.mp4", "sav_048780.mp4",
    "sav_048830.mp4", "sav_049007.mp4", "sav_026462.mp4", "sav_024477.mp4",
    "sav_005448.mp4", "sav_035332.mp4", "sav_003834.mp4", "sav_037935.mp4",
    "sav_032656.mp4", "sav_015370.mp4",
}

local_rank = None


class SkipSample(Exception):
    """Signal that this sample should be skipped (e.g., empty masks)."""
    pass


def _safe_rank():
    try:
        if "RANK" in os.environ:
            return int(os.environ["RANK"])
        if "LOCAL_RANK" in os.environ:
            return int(os.environ["LOCAL_RANK"])
        import torch.distributed as dist
        if dist.is_available() and dist.is_initialized():
            return dist.get_rank()
    except Exception:
        pass
    return 0


def rank0_print(*args):
    if local_rank == 0:
        print(*args)


def read_jsonl(path):
    with open(path, "r") as f:
        return [json.loads(line) for line in f]


def return_gpt_conv_index(convs):
    gpt_idx = None
    for ii, c in enumerate(convs):
        role = c.get("from") or c.get("role")
        if role in ("gpt", "assistant"):
            gpt_idx = ii
            break
    return gpt_idx


def _safe_get_obj_id(token: str) -> Optional[int]:
    """Extract the id from a key like 'object_17' or 'object 17'; return None on failure."""
    m = OBJ_ID_RE.search(token)
    if not m:
        return None
    try:
        return int(m.group(1))
    except Exception:
        return None


def _filter_objects(objects_list: List[dict], keep_ids: Set[int]) -> List[dict]:
    """Keep only the entries of ``objects_list`` whose id is in ``keep_ids``.

    ``objects_list`` looks like ``[{"object_0": "sedan", "attributes": [...]}, ...]``;
    order and structure are preserved.
    """
    new_objects = []
    for obj in objects_list:
        if not isinstance(obj, dict):
            continue
        key_candidates = [k for k in obj.keys() if isinstance(k, str) and k.startswith("object_")]
        if not key_candidates:
            continue
        oid = _safe_get_obj_id(key_candidates[0])
        if oid is not None and oid in keep_ids:
            new_objects.append(obj)
    return new_objects


def preprocess_qwen_2_visual(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    grid_thw_image: List = [],
    grid_thw_video: List = [],
) -> Dict:
    roles = {"human": "user", "gpt": "assistant"}
    system_message = "You are a helpful assistant."

    tokenizer = copy.deepcopy(tokenizer)
    chat_template = "{% for message in messages %}{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"
    tokenizer.chat_template = chat_template

    visual_replicate_index_image = 0
    visual_replicate_index_video = 0
    input_ids, targets = [], []

    for i, source in enumerate(sources):
        try:
            if roles[source[0]["from"]] != roles["human"]:
                source = source[1:]
        except:
            print(sources)

        input_id, target = [], []

        # return_dict=False keeps the plain id-list output across transformers 4.x / 5.x.
        input_id += list(tokenizer.apply_chat_template(
            [{"role": "system", "content": system_message}], return_dict=False
        ))
        target += [IGNORE_INDEX] * len(input_id)

        for conv in source:
            try:
                role = conv["role"]
                content = conv["content"]
            except:
                role = conv["from"]
                content = conv["value"]

            role = roles.get(role, role)
            if role == "user":
                if "<image>" in content:
                    parts = content.split("<image>")
                    new_parts = []
                    for i in range(len(parts) - 1):
                        new_parts.append(parts[i])
                        replacement = (
                            "<|vision_start|>"
                            + f"<|image_pad|>"
                            * grid_thw_image[visual_replicate_index_image]
                            + "<|vision_end|>"
                        )
                        new_parts.append(replacement)
                        visual_replicate_index_image += 1
                    new_parts.append(parts[-1])
                    content = "".join(new_parts)

                if "<video>" in content:
                    parts = content.split("<video>")
                    new_parts = []
                    for i in range(len(parts) - 1):
                        new_parts.append(parts[i])
                        replacement = (
                            "<|vision_start|>"
                            + f"<|video_pad|>"
                            * grid_thw_video[visual_replicate_index_video]
                            + "<|vision_end|>"
                        )
                        new_parts.append(replacement)
                        visual_replicate_index_video += 1
                    new_parts.append(parts[-1])
                    content = "".join(new_parts)

            conv = [{"role": role, "content": content}]
            encode_id = list(tokenizer.apply_chat_template(conv, return_dict=False))
            input_id += encode_id
            if role in ["user", "system"]:
                target += [IGNORE_INDEX] * len(encode_id)
            else:
                target_mask = encode_id.copy()
                target_mask[:3] = [IGNORE_INDEX] * 3
                target += target_mask

        assert len(input_id) == len(target), f"{len(input_id)} != {len(target)}"
        input_ids.append(input_id)
        targets.append(target)

    input_ids = torch.tensor(input_ids, dtype=torch.long)
    targets = torch.tensor(targets, dtype=torch.long)

    return dict(
        input_ids=input_ids,
        labels=targets,
    )


class LazySupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning on videos with object trajectories.

    Each annotation entry provides:
        video:              path to the video file
        mask_json:          per-frame, per-object RLE segmentation masks (COCO format)
        obj_list:           object indices into each frame's mask list
        has_relationships:  whether the target scene graph contains relationships
        conversations:      user prompt + assistant target (JSON scene graph)
    """

    def __init__(self, tokenizer: transformers.PreTrainedTokenizer, data_args):
        super(LazySupervisedDataset, self).__init__()

        dataset = data_args.dataset_use.split(",")
        dataset_list = data_list(dataset)
        rank0_print(f"Loading datasets: {dataset_list}")

        list_data_dict = []

        for data in dataset_list:
            file_format = data["annotation_path"].split(".")[-1]
            if file_format == "jsonl":
                annotations = read_jsonl(data["annotation_path"])
            else:
                annotations = json.load(open(data["annotation_path"], "r"))
            sampling_rate = data.get("sampling_rate", 1.0)
            if sampling_rate < 1.0:
                rng = random.Random(42)
                annotations = rng.sample(
                    annotations, int(len(annotations) * sampling_rate)
                )
                print(f"sampling {len(annotations)} examples from dataset {data}")
            else:
                rank0_print(f"dataset name: {data}")
            for ann in annotations:
                ann["data_path"] = data["data_path"]
            list_data_dict += annotations

        rank0_print(f"Total training samples: {len(list_data_dict)}")

        rank0_print("Formatting inputs...Skip in lazy mode")
        self.tokenizer = tokenizer
        self.list_data_dict = list_data_dict
        self.data_args = data_args
        self.data_args.processor.video_processor.max_pixels = data_args.max_pixels
        self.data_args.processor.video_processor.min_pixels = data_args.min_pixels

    def __len__(self):
        return len(self.list_data_dict)

    @staticmethod
    def _is_skipped_video(video_path: str) -> bool:
        return os.path.basename(video_path) in SKIP_VIDEO_FILENAMES

    @property
    def lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            img_tokens = 128 if "image" in sample else 0
            length_list.append(
                sum(len(conv["value"].split()) for conv in sample["conversations"])
                + img_tokens
            )
        return length_list

    @property
    def modality_lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            cur_len = sum(
                len(conv["value"].split()) for conv in sample["conversations"]
            )
            cur_len = (
                cur_len if ("image" in sample) or ("video" in sample) else -cur_len
            )
            length_list.append(cur_len)
        return length_list

    def process_video(self, video_file):
        max_decord_attempts = 3
        for attempt in range(max_decord_attempts):
            try:
                return self.video_decord(video_file)
            except Exception as e:
                print(f"Decord attempt {attempt + 1} failed: {e}")
        return self.video_torchcodec(video_file)

    def video_decord(self, video_file):
        if not os.path.exists(video_file):
            print(f"File not exist: {video_file}")
        vr = VideoReader(video_file, num_threads=4)
        total_frames = len(vr)
        avg_fps = vr.get_avg_fps()
        video_length = total_frames / avg_fps
        interval = getattr(self.data_args, "base_interval", 1)

        num_frames_to_sample = round(video_length / interval)
        video_min_frames = getattr(self.data_args, "video_min_frames", 4)
        video_max_frames = getattr(self.data_args, "video_max_frames", 128)

        target_frames = min(
            max(num_frames_to_sample, video_min_frames), video_max_frames
        )
        frame_idx = np.linspace(0, total_frames - 1, target_frames, dtype=int)
        frame_idx = np.unique(frame_idx)
        video = vr.get_batch(frame_idx).asnumpy()
        return self.process_video_frames(video, frame_idx, video_length)

    def video_torchcodec(self, video_file):
        # Imported here so that torchcodec (and its FFmpeg shared libraries) is only needed
        # when decord has failed on a file.
        from torchcodec.decoders import VideoDecoder

        device = "cpu"
        decoder = VideoDecoder(video_file, device=device)
        total_frames = decoder.metadata.num_frames
        avg_fps = decoder.metadata.average_fps
        video_length = total_frames / avg_fps
        interval = getattr(self.data_args, "base_interval", 1)

        num_frames_to_sample = round(video_length / interval)
        video_min_frames = getattr(self.data_args, "video_min_frames", 4)
        video_max_frames = getattr(self.data_args, "video_max_frames", 128)

        target_frames = min(
            max(num_frames_to_sample, video_min_frames), video_max_frames
        )
        frame_idx = np.linspace(0, total_frames - 1, target_frames, dtype=int)
        frame_idx = np.unique(frame_idx)
        frame_batch = decoder.get_frames_at(indices=frame_idx.tolist())
        video = frame_batch.data.cpu().numpy()
        return self.process_video_frames(video, frame_idx, video_length)

    def process_video_frames(self, video, frame_idx, video_length):
        fps = len(frame_idx) / video_length
        processor = copy.deepcopy(self.data_args.processor.video_processor)
        # Max / min pixels within a frame.
        processor.max_pixels = self.data_args.video_max_frame_pixels
        processor.min_pixels = self.data_args.video_min_frame_pixels
        processor.size["longest_edge"] = processor.max_pixels
        processor.size["shortest_edge"] = processor.min_pixels
        video_processed = processor.preprocess(
            videos=video, return_tensors="pt"
        )
        video_tensor = video_processed["pixel_values_videos"]
        video_grid_thw = video_processed["video_grid_thw"][0]
        second_per_grid_ts = [
            self.data_args.processor.video_processor.temporal_patch_size / fps
        ] * len(video_processed["video_grid_thw"])
        return video_tensor, video_grid_thw, frame_idx.tolist(), second_per_grid_ts

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        num_base_retries = 3
        num_final_retries = 30
        # Try the requested sample first.
        for attempt_idx in range(num_base_retries):
            try:
                while self._is_skipped_video(self.list_data_dict[i]["video"]):
                    print(f"[rank = {_safe_rank()}] Sample {i} - {self.list_data_dict[i]['video']} is in the skip list, moving to the next sample")
                    i = (i + 1) % len(self.list_data_dict)
                sample = self._get_item(i)
                return sample
            except SkipSample as e:
                print(f"1. [rank = {_safe_rank()}] [Try #{attempt_idx}] Skip sample {i} - {self.list_data_dict[i]['video']}: {e}")
                continue
            except Exception as e:
                # Sleep in case it is a transient storage issue.
                print(f"2. [rank = {_safe_rank()}] [Try #{attempt_idx}] Failed to fetch sample {i} - {self.list_data_dict[i]['video']}. Exception:", e)
                time.sleep(1)

        # Try the next sample, in case the failure is caused by file corruption.
        for attempt_idx in range(num_base_retries):
            try:
                next_index = min(i + 1, len(self.list_data_dict) - 1)
                sample = self._get_item(next_index)
                return sample
            except SkipSample as e:
                print(f"3. [rank = {_safe_rank()}] [Try other #{attempt_idx}] Skip sample {next_index} - {self.list_data_dict[next_index]['video']}: {e}")
                continue
            except Exception as e:
                print(
                    f"4. [rank = {_safe_rank()}] [Try other #{attempt_idx}] Failed to fetch sample {next_index} - {self.list_data_dict[next_index]['video']}. Exception:",
                    e,
                )
                pass

        # Fall back to random samples.
        for attempt_idx in range(num_final_retries):
            try:
                rand_index = random.randrange(len(self.list_data_dict) - 1)
                print(
                    f"5. [rank = {_safe_rank()}] [Try random sample {rand_index} - {self.list_data_dict[rand_index]['video']}",
                )
                sample = self._get_item(rand_index)
                return sample
            except SkipSample as e:
                print(f"6. [rank = {_safe_rank()}] [Try random #{attempt_idx}] Skip sample {rand_index} - {self.list_data_dict[rand_index]['video']}: {e}")
                continue
            except Exception:
                continue

        sample = self._get_item(i)
        return sample

    def _renumber_objects_and_relationships(self, convs, start_idx: int = 1):
        """Renumber the assistant target after mask filtering.

        Object keys ``object_{orig_id}`` become consecutive ``object {k}`` (k starting at
        ``start_idx``), matching the "Object k: " labels inserted in front of each object's
        visual tokens. Relationship triplets ``[sub_id, predicate, obj_id, windows]`` are
        remapped to the new numbering.
        """
        gpt_idx = return_gpt_conv_index(convs)
        assert gpt_idx is not None, "No gpt/assistant message found"
        key = "value" if "value" in convs[gpt_idx] else ("content" if "content" in convs[gpt_idx] else None)
        assert key is not None, "No gpt/assistant message content found"

        gpt_dict = json.loads(convs[gpt_idx][key])
        assert isinstance(gpt_dict, dict), "The gpt/assistant message is not a JSON object"

        id_map = {}  # original id -> new numeric id
        new_objects = []
        for idx, obj in enumerate(gpt_dict["objects"], start=start_idx):
            old_key = list(obj.keys())[0]  # e.g. 'object_17'
            old_id = int(old_key.split("_")[1])
            id_map[old_id] = idx
            new_obj = {f"object {idx}": obj[old_key]}
            if "attributes" in obj:
                new_obj["attributes"] = obj["attributes"]
            new_objects.append(new_obj)
        gpt_dict["objects"] = new_objects

        if "relationships" in gpt_dict:
            new_relationships = []
            for rel in gpt_dict["relationships"]:  # e.g. [17, 'wearing', 20, [[0, 11]]]
                sub_id, predicate, obj_id = rel[0], rel[1], rel[2]
                new_sub = CAMERA_ID if sub_id == CAMERA_ID else id_map.get(sub_id)
                new_obj = CAMERA_ID if obj_id == CAMERA_ID else id_map.get(obj_id)
                new_relationships.append([new_sub, predicate, new_obj] + rel[3:])
            gpt_dict["relationships"] = new_relationships

        convs[gpt_idx][key] = json.dumps(gpt_dict, ensure_ascii=False)
        return id_map

    def _get_item(self, i) -> Dict[str, torch.Tensor]:
        sources = self.list_data_dict[i]
        if isinstance(i, int):
            sources = [sources]
        assert len(sources) == 1, "Don't know why it is wrapped to a list"  # FIXME

        # ---- (1) Decode and preprocess the video ----
        video_file = os.path.join(self.list_data_dict[i]["data_path"], self.list_data_dict[i]["video"])
        video, video_grid_thw, sampled_frame_idx, second_per_grid_ts = self.process_video(video_file)
        video = [video]

        video_grid_thw_merged = copy.deepcopy(video_grid_thw)
        if not isinstance(video_grid_thw, Sequence):
            video_grid_thw_merged = [video_grid_thw_merged]
            video_grid_thw = [video_grid_thw]
        video_grid_thw_merged = [
            merged_thw.prod() // self.data_args.processor.video_processor.merge_size**2
            for merged_thw in video_grid_thw_merged
        ]

        # ---- (2) Build per-object mask tensors on the resized frame grid ----
        # Work on a copy: _get_item must not mutate the cached annotation, so that
        # retries and later epochs see the original sample.
        convs = copy.deepcopy(sources[0]["conversations"])
        mask_json_path = self.list_data_dict[i]["mask_json"]
        with open(mask_json_path, "r") as f:
            mask_data = json.load(f)  # list(frames) of list(objects) of {"size": [H, W], "counts": ...}

        obj_list = self.list_data_dict[i].get("obj_list", [])
        assert len(obj_list) > 0

        sampled_idx = sampled_frame_idx[0] if isinstance(sampled_frame_idx[0], list) else sampled_frame_idx

        g0 = video_grid_thw[0]  # tensor([T, H, W])
        T, H_patch, W_patch = int(g0[0]), int(g0[1]), int(g0[2])
        patch_size = self.data_args.processor.video_processor.patch_size
        H_rz, W_rz = H_patch * patch_size, W_patch * patch_size

        O = len(obj_list)
        N = len(sampled_idx)
        obj_masks = torch.zeros((O, N, H_rz, W_rz), dtype=torch.float32)
        for o_idx, oid in enumerate(obj_list):
            for n_idx, fidx in enumerate(sampled_idx):
                if fidx < 0 or fidx >= len(mask_data):
                    continue
                frame_objs = mask_data[fidx]
                if frame_objs is None or oid < 0 or oid >= len(frame_objs):
                    continue
                rle = frame_objs[oid]
                if not rle:
                    continue
                # An object absent from a frame is stored as an all-zero RLE. Skip those
                # without decoding: some released placeholders encode a run shorter than
                # size[0] * size[1], and pycocotools then leaves the rest of its output
                # buffer uninitialised, so decoding them yields garbage rather than zeros.
                # area() reads the run lengths only, so it is both safe and cheap.
                if maskUtils.area({"size": rle["size"], "counts": rle["counts"]}) == 0:
                    continue
                m = maskUtils.decode({"size": rle["size"], "counts": rle["counts"]})
                if m.ndim == 3:
                    m = m[:, :, 0]
                m_t = torch.from_numpy(m.astype(np.uint8)).unsqueeze(0).unsqueeze(0).float()
                m_rz = F.interpolate(m_t, size=(H_rz, W_rz), mode="nearest")[0, 0]
                obj_masks[o_idx, n_idx] = (m_rz > 0.5).to(torch.float32)

        # ---- (3) Drop objects whose masks are empty on the sampled frames, and keep the
        #          assistant target consistent with the surviving objects ----
        keep_idx = (obj_masks.view(O, -1).sum(dim=1) > 0).nonzero(as_tuple=False).squeeze(1).tolist()
        if len(keep_idx) < O:
            if len(keep_idx) == 0:
                raise SkipSample("All objects have empty masks")
            obj_masks = obj_masks[keep_idx]
            new_obj_list = [obj_list[j] for j in keep_idx]

            gpt_idx = return_gpt_conv_index(convs)
            if gpt_idx is not None:
                key = "value" if "value" in convs[gpt_idx] else ("content" if "content" in convs[gpt_idx] else None)
                if key is not None and isinstance(convs[gpt_idx][key], str):
                    gpt_dict = json.loads(convs[gpt_idx][key])
                    keep_set = set(new_obj_list)
                    if "objects" in gpt_dict and isinstance(gpt_dict["objects"], list):
                        gpt_dict["objects"] = _filter_objects(gpt_dict["objects"], keep_set)
                    if self.list_data_dict[i]["has_relationships"] and "relationships" in gpt_dict:
                        gpt_dict["relationships"] = [
                            rel for rel in gpt_dict["relationships"]
                            if (rel[0] == CAMERA_ID or rel[0] in keep_set)
                            and (rel[2] == CAMERA_ID or rel[2] in keep_set)
                        ]
                    convs[gpt_idx][key] = json.dumps(gpt_dict, ensure_ascii=False)

        self._renumber_objects_and_relationships(convs, start_idx=1)

        # ---- (4) Tokenize the conversation ----
        data_dict = preprocess_qwen_2_visual(
            [convs],
            self.tokenizer,
            grid_thw_image=None,
            grid_thw_video=video_grid_thw_merged,
        )

        data_dict["obj_masks"] = obj_masks.to(torch.uint8).cpu()
        data_dict["pixel_values_videos"] = torch.cat(video, dim=0)
        data_dict["video_grid_thw"] = torch.cat(
            [thw.unsqueeze(0) for thw in video_grid_thw], dim=0
        )
        data_dict["second_per_grid_ts"] = torch.tensor(second_per_grid_ts)
        data_dict["video_name"] = self.list_data_dict[i].get("video", "N/A")
        return data_dict


@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels, obj_masks, second_per_grid_ts, video_names = tuple(
            [instance[key] for instance in instances]
            for key in ("input_ids", "labels", "obj_masks", "second_per_grid_ts", "video_name")
        )
        second_per_grid_ts = torch.concat(second_per_grid_ts, dim=0)
        input_ids = [ids.squeeze(0) for ids in input_ids]
        labels = [ids.squeeze(0) for ids in labels]
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(
            labels, batch_first=True, padding_value=IGNORE_INDEX
        )
        input_ids = input_ids[:, : self.tokenizer.model_max_length]
        labels = labels[:, : self.tokenizer.model_max_length]
        batch = dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
            second_per_grid_ts=second_per_grid_ts,
        )

        videos = list(
            instance["pixel_values_videos"]
            for instance in instances
            if "pixel_values_videos" in instance
        )
        if len(videos) != 0:
            concat_videos = torch.cat([video for video in videos], dim=0)
            video_grid_thw = [
                instance["video_grid_thw"]
                for instance in instances
                if "video_grid_thw" in instance
            ]
            video_grid_thw = torch.cat(video_grid_thw, dim=0)
        else:
            concat_videos = None
            video_grid_thw = None

        batch["pixel_values_videos"] = concat_videos
        batch["video_grid_thw"] = video_grid_thw

        batch["position_ids"] = None
        batch["obj_masks"] = [t.cpu().to(torch.uint8) for t in obj_masks]
        batch["video_name"] = video_names
        return batch


def make_supervised_data_module(
    tokenizer: transformers.PreTrainedTokenizer, data_args
) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    train_dataset = LazySupervisedDataset(tokenizer=tokenizer, data_args=data_args)
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    return dict(
        train_dataset=train_dataset, eval_dataset=None, data_collator=data_collator
    )
