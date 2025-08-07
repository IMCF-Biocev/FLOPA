# ptuio/marker.py

import numpy as np
from typing import Dict, Union, Tuple
from .reconstructor import ScanConfig
# AS OF NOW, THIS SCRIPT IS NOT NEEDED. DISSOLVE THE USEFUL PARTS IN RECONSTRUCTOR OR UTILS (NOT EXISTING YET)

# class MarkerInterpreter:
    # def __init__(self, marker_map: Dict[Union[int, Tuple[int, ...]], str]):
    #     """
    #     Initializes the interpreter with a user-defined marker map.
    #     Keys can be individual integers or tuples of integers representing multiple codes.
    #     Values are the semantic marker labels (e.g., 'line_start').
    #     """
    #     self.marker_map = self._normalize(marker_map)

    # def _normalize(self, raw_map: Dict) -> Dict[Tuple[int, ...], str]:
    #     result = {}
    #     for k, v in raw_map.items():
    #         if isinstance(k, int):
    #             result[(k,)] = v
    #         elif isinstance(k, (list, tuple, set)):
    #             result[tuple(k)] = v
    #         else:
    #             raise TypeError(f"Invalid marker key: {k}")
    #     return result

    # def classify(self, events: np.ndarray) -> np.ndarray:
    #     """
    #     Classifies marker events based on the defined map.

    #     Parameters:
    #         events: structured array with 'channel' and 'special' fields

    #     Returns:
    #         An array of marker labels (str), empty string if no match
    #     """
    #     is_marker = (events['channel'] < 63) & (events['special'] != 0)
    #     labels = np.full(events.shape[0], '', dtype=object)

    #     for codes, label in self.marker_map.items():
    #         mask = is_marker & np.isin(events['channel'], codes)
    #         labels[mask] = label

    #     return labels

    # def extract(self, events: np.ndarray, label: str) -> np.ndarray:
    #     """
    #     Extracts events matching the given marker label.
    #     """
    #     classified = self.classify(events)
    #     return events[classified == label]



# class MarkerRoles:
#     def __init__(
#         self,
#         frame_start_channel: Optional[Union[int, Tuple[int, ...]]] = None,
#         forward_line_channel: Optional[Union[int, Tuple[int, ...]]] = None,
#         backward_line_channel: Optional[Union[int, Tuple[int, ...]]] = None,
#     ):
#         self._role_map = {}

#         for name, codes in {
#             "frame_start": frame_start_channel,
#             "forward_line": forward_line_channel,
#             "backward_line": backward_line_channel,
#         }.items():
#             if codes is None:
#                 continue
#             if isinstance(codes, int):
#                 codes = [codes]
#             elif isinstance(codes, tuple):
#                 codes = list(codes)
#             elif not isinstance(codes, list):
#                 raise TypeError(f"{name}_code must be int, tuple, or list.")
#             self._role_map[name] = codes

#     def get_codes(self, role: str) -> List[int]:
#         return self._role_map.get(role, [])

#     def __getitem__(self, role: str) -> List[int]:
#         return self.get_codes(role)

#     def __contains__(self, role: str) -> bool:
#         return role in self._role_map

#     def roles(self) -> List[str]:
#         return list(self._role_map.keys())

#     def to_dict(self) -> dict:
#         return dict(self._role_map)

#     def __repr__(self):
#         return f"MarkerRoles({self._role_map})"


# class MarkerRoles:
#     def __init__(
#         self,
#         frame_start: Union[int, Tuple[int, ...]],
#         forward_line_start: Union[int, Tuple[int, ...]],
#         backward_line_start: Union[int, Tuple[int, ...]]

#     ):
#         self.frame_start = (frame_start,) if isinstance(frame_start, int) else tuple(frame_start)
#         self.forward_line_start = (forward_line_start,) if isinstance(forward_line_start, int) else tuple(forward_line_start)
#         self.backward_line_start = (backward_line_start,) if isinstance(backward_line_start, int) else tuple(backward_line_start)


#     def to_dict(self) -> dict:
#         return {
#             "frame_start": self.frame_start,
#             "forward_line_start": self.forward_line_start,
#             "backward_line_start": self.backward_line_start
#         }

#     @classmethod
#     def from_dict(cls, d: dict) -> "MarkerRoles":
#         return cls(
#             frame_start=d.get("frame_start", ()),
#             line_start=d.get("line_start", ())
#         )

#     def __repr__(self):
#         return f"MarkerRoles(frame_start={self.frame_start}, line_start={self.line_start})"


# class MarkerInterpreter:
#     def __init__(self, marker_definitions: list[dict]):
#         """
#         Each dict should contain:
#             - role: str, e.g., "frame_start"
#             - label: str, e.g., "line_marker_bwd"
#             - code: int, e.g., 2
#         """
#         self._role_to_label = {}
#         self._role_to_codes = {}
#         self._code_to_label = {}

#         for entry in marker_definitions:
#             role = entry["role"]
#             label = entry["label"]
#             codes = entry["codes"]  

#             if isinstance(codes, int):
#                 codes = [codes]

#             if role in self._role_to_codes:
#                 raise ValueError(f"Duplicate role: {role}")
#             self._role_to_codes[role] = codes
#             self._role_to_label[role] = label

#             for code in codes:
#                 if code in self._code_to_label:
#                     raise ValueError(f"Duplicate special code: {code}")
#                 self._code_to_label[code] = label

#     def classify(self, events):
#         is_marker = (events['channel'] < 63) & (events['special'] != 0)
#         labels = np.full(events.shape[0], '', dtype=object)
#         for code, label in self._code_to_label.items():
#             mask = is_marker & (events['special'] == code)
#             labels[mask] = label
#         return labels
    
#     def extract_by_role(self, events, role):
#         codes = self._role_to_codes[role]
#         return events[(events['channel'] < 63) & np.isin(events['special'], codes)]

#     def available_roles(self):
#         return list(self._role_to_label.keys())

#     def label_for_role(self, role):
#         return self._role_to_label[role]

#     def code_for_role(self, role):
#         label = self._role_to_label[role]
#         return self._label_to_code[label]



# class MarkerInterpreter: 
#     # MAKE OVER ENTIRELY. MAKE IT ACCEPT TUPPLE AND BE CALLED 
#     # cfg = ScanConfig(bidirectional=True, line_start_marker=1, line_stop_marker=2, frame_start_marker=4)
#     # interpreter = MarkerInterpreter(scan_config=cfg)
#     # line_start_events = interpreter.extract(corrected, cfg.line_start_marker)
#     #  i.e., tuple
#     def __init__(self, scan_config: ScanConfig):
#         if not isinstance(scan_config, ScanConfig):
#             raise TypeError("MarkerInterpreter requires a ScanConfig object")
#         self._role_to_channel = {
#             "frame_start": scan_config.frame_start_marker,
#             "line_start": scan_config.line_start_marker,
#             "line_stop": scan_config.line_stop_marker,
#         }

#         # Invert for special → role lookup (optional, if needed for classify)
#         self._channel_to_role = {
#             code: role
#             for role, codes in self._role_to_channel.items()
#             for code in codes
#         }

#     def extract_by_role(self, events, role: str):
#         if role not in self._role_to_channel:
#             raise ValueError(f"Unknown role: {role}")
#         codes = self._role_to_channel[role]
#         return events[
#             (events['channel'] < 63) & np.isin(events['special'], codes)
#         ]

#     def classify(self, events):
#         is_marker = (events['channel'] < 63) & (events['special'] != 0)
#         labels = np.full(events.shape[0], '', dtype=object)
#         for channel, role in self._channel_to_role.items():
#             mask = is_marker & (events['special'] == channel)
#             labels[mask] = role
#         return labels

#     def roles(self):
#         return list(self._role_to_channel.keys())

#     def channel_for(self, role):
#         return self._role_to_channel.get(role, ())


class MarkerInterpreter: 
    # MAKE OVER ENTIRELY. MAKE IT ACCEPT TUPPLE AND BE CALLED 
    # cfg = ScanConfig(bidirectional=True, line_start_marker=1, line_stop_marker=2, frame_start_marker=4)
    # interpreter = MarkerInterpreter(scan_config=cfg)
    # line_start_events = interpreter.extract(corrected, cfg.line_start_marker)
    #  i.e., tuple

    def __init__(self, scan_config: ScanConfig):
        if not isinstance(scan_config, ScanConfig):
            raise TypeError("MarkerInterpreter requires a ScanConfig object")
        self.scan_config = scan_config

        self._role_to_codes = {
            "frame_start": self.scan_config.frame_start_marker,
        }

        # Invert for special → role lookup (optional, if needed for classify)
        self._channel_to_role = {
            code: role
            for role, codes in self._role_to_channel.items()
            for code in codes
        }

    # def extract_by_role(self, events, role: str):
    #     if role not in self._role_to_channel:
    #         raise ValueError(f"Unknown role: {role}")
    #     codes = self._role_to_channel[role]
    #     return events[
    #         (events['channel'] < 63) & np.isin(events['special'], codes)
    #     ]

    def classify(self, events):
        is_marker = (events['channel'] < 63) & (events['special'] != 0)
        labels = np.full(events.shape[0], '', dtype=object)
        for channel, role in self._channel_to_role.items():
            mask = is_marker & (events['special'] == channel)
            labels[mask] = role
        return labels

    def roles(self):
        return list(self._role_to_channel.keys())

    def channel_for(self, role):
        return self._role_to_channel.get(role, ())


# --- Optional Helpers ---

def marker_events(events: np.ndarray) -> np.ndarray:
    """Return only events where channel == 63 and special != 15 (non-overflow markers)."""
    return events[(events['channel'] < 63) & (events['special'] != 0)]

def overflow_events(events: np.ndarray) -> np.ndarray:
    """Return overflow marker events."""
    return events[(events['channel'] == 63) & (events['special'] != 0)]

def get_marker_distribution(events: np.ndarray) -> Dict[int, int]:
    """Returns a count of each special marker code."""
    mask = (events['channel'] < 63) & (events['special'] != 0)
    markers = events['channel'][mask]
    unique, counts = np.unique(markers, return_counts=True)
    return dict(zip(unique.tolist(), counts.tolist()))

