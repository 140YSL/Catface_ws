import os
import tempfile
from pathlib import Path

import numpy as np


AU_DESCRIPTION = [
    "左眼闭眼",
    "右眼闭眼",
    "皱鼻+上唇提升",
    "唇部前撅",
    "下颌下垂",
    "左耳向前",
    "左耳旋转",
    "左耳向后",
    "右耳向前",
    "右耳旋转",
    "右耳向后",
]

AU_NAMES = [
    "LAU143",
    "RAU143",
    "AU109+110",
    "AU118",
    "AU26",
    "LEAD101",
    "LEAD104",
    "LEAD106",
    "READ101",
    "READ104",
    "READ106",
]


class AnnotationStore:
    def __init__(self, path, records=None):
        self.path = Path(path)
        self._records = records or {}

    @classmethod
    def load(cls, path):
        path = Path(path)
        if not path.exists():
            return cls(path)

        with np.load(path, allow_pickle=False) as data:
            required_keys = {"frame_indices", "time", *AU_NAMES}
            legacy_keys = {"frame_indices", "timestamps", "au_intensities"}
            is_legacy = legacy_keys.issubset(data.files)
            missing_keys = required_keys.difference(data.files)
            if missing_keys and not is_legacy:
                raise ValueError(
                    "NPZ 缺少字段: " + ", ".join(sorted(missing_keys))
                )

            frame_indices = np.asarray(data["frame_indices"], dtype=np.int64)
            if is_legacy:
                timestamps = np.asarray(data["timestamps"], dtype=np.float64)
                intensities = np.asarray(data["au_intensities"])
            else:
                timestamps = np.asarray(data["time"], dtype=np.float64)
                intensities = np.column_stack(
                    [data[au_name] for au_name in AU_NAMES]
                )

        if frame_indices.ndim != 1 or timestamps.shape != frame_indices.shape:
            raise ValueError("NPZ 中帧索引或时间戳的维度不正确")
        if intensities.shape != (len(frame_indices), len(AU_DESCRIPTION)):
            raise ValueError(
                f"NPZ 中 AU 数组应为 (标注帧数, {len(AU_DESCRIPTION)})"
            )
        if np.any(frame_indices < 0):
            raise ValueError("NPZ 中存在负数帧索引")
        if len(np.unique(frame_indices)) != len(frame_indices):
            raise ValueError("NPZ 中存在重复帧索引")
        if not np.all(np.isfinite(timestamps)) or np.any(timestamps < 0):
            raise ValueError("NPZ 中存在无效时间戳")
        if not np.issubdtype(intensities.dtype, np.integer):
            raise ValueError("NPZ 中 AU 强度必须为整数")
        if np.any(intensities < 0) or np.any(intensities > 5):
            raise ValueError("NPZ 中 AU 强度必须位于 0 到 5 之间")

        records = {
            int(frame_index): (
                float(timestamp),
                intensity_row.astype(np.uint8, copy=True),
            )
            for frame_index, timestamp, intensity_row in zip(
                frame_indices, timestamps, intensities
            )
        }
        return cls(path, records)

    def get(self, frame_index):
        record = self._records.get(int(frame_index))
        if record is None:
            return None
        timestamp, intensities = record
        return timestamp, intensities.copy()

    def get_latest_before(self, frame_index):
        frame_index = int(frame_index)
        candidates = [index for index in self._records if index <= frame_index]
        if not candidates:
            return None
        timestamp, intensities = self._records[max(candidates)]
        return timestamp, intensities.copy()

    def set(self, frame_index, timestamp, intensities):
        values = np.asarray(intensities, dtype=np.uint8)
        if values.shape != (len(AU_DESCRIPTION),):
            raise ValueError(f"需要 {len(AU_DESCRIPTION)} 个 AU 强度值")
        if np.any(values > 5):
            raise ValueError("AU 强度必须位于 0 到 5 之间")
        if frame_index < 0 or not np.isfinite(timestamp) or timestamp < 0:
            raise ValueError("帧索引和时间戳必须为非负数")

        self._records[int(frame_index)] = (float(timestamp), values.copy())

    def save(self):
        frame_indices = sorted(self._records)
        timestamps = np.asarray(
            [self._records[index][0] for index in frame_indices],
            dtype=np.float64,
        )
        intensities = np.asarray(
            [self._records[index][1] for index in frame_indices],
            dtype=np.uint8,
        ).reshape(-1, len(AU_DESCRIPTION))

        temporary_path = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="wb",
                dir=self.path.parent,
                prefix=self.path.stem + ".",
                suffix=".tmp",
                delete=False,
            ) as temporary_file:
                temporary_path = Path(temporary_file.name)
                au_values = {
                    au_name: intensities[:, column]
                    for column, au_name in enumerate(AU_NAMES)
                }
                np.savez_compressed(
                    temporary_file,
                    frame_indices=np.asarray(frame_indices, dtype=np.int64),
                    time=timestamps,
                    au_names=np.asarray(AU_NAMES, dtype=np.str_),
                    au_descriptions=np.asarray(AU_DESCRIPTION, dtype=np.str_),
                    **au_values,
                )
            os.replace(temporary_path, self.path)
        finally:
            if temporary_path is not None and temporary_path.exists():
                temporary_path.unlink()

    def __len__(self):
        return len(self._records)
