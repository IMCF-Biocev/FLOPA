# ptuio/file.py
import os
import struct
from enum import IntEnum
from typing import BinaryIO
from datetime import datetime, timedelta
import numpy as np


class PqFileType(IntEnum):
    PTU = 0x005254545051  # 'PQTTTR\0\0' as int
    PHU = 0x005453494851  # 'PQHISTO\0'

class TagTypes:
    tyEmpty8      = 0xFFFF0008
    tyBool8       = 0x00000008
    tyInt8        = 0x10000008
    tyBitSet64    = 0x11000008
    tyColor8      = 0x12000008
    tyFloat8      = 0x20000008
    tyTDateTime   = 0x21000008
    tyFloat8Array = 0x2001FFFF
    tyAnsiString  = 0x4001FFFF
    tyWideString  = 0x4002FFFF
    tyBinaryBlob  = 0xFFFFFFFF

def read_tag_value(file, tag_type):
    match tag_type:
        case TagTypes.tyEmpty8:
            file.read(8)  # skip 8 bytes
            return "<Empty>"

        case TagTypes.tyBool8:
            value = struct.unpack('q', file.read(8))[0]
            return bool(value)

        case TagTypes.tyInt8 | TagTypes.tyBitSet64 | TagTypes.tyColor8:
            return struct.unpack('q', file.read(8))[0]

        case TagTypes.tyFloat8:
            return struct.unpack('d', file.read(8))[0]

        case TagTypes.tyTDateTime:
            raw = struct.unpack('d', file.read(8))[0]
            date = datetime(1899, 12, 30) + timedelta(days=raw)
            return date

        case TagTypes.tyFloat8Array:
            length = struct.unpack('q', file.read(8))[0]
            file.seek(length, 1)  # skip the array data
            return f"<Float array with {length // 8} Entries>"

        case TagTypes.tyAnsiString | TagTypes.tyWideString:
            length = struct.unpack('q', file.read(8))[0]
            string_data = file.read(length)
            clean_string = string_data.replace(b'\x00', b'').decode('utf-8', errors='ignore')
            return clean_string

        case TagTypes.tyBinaryBlob:
            length = struct.unpack('q', file.read(8))[0]
            file.seek(length, 1)  # skip binary blob
            return f"<Binary Blob with {length} Bytes>"

        case _:
            raise ValueError(f"Unknown tag type: {hex(tag_type)}")


class Header:
    def __init__(self, path: str):
        self.path = path
        self.tags = {}
        self.file_type = None
        self.version = None
        self.record_type = None
        self.bits_per_record = 32  # default for TTTR
        self.data_offset = 0

        with open(path, 'rb') as f:
            self._read_header(f)

    def _read_header(self, f: BinaryIO):
        magic = f.read(8)
        if magic.startswith(b'PQTTTR'):
            self.file_type = PqFileType.PTU
        elif magic.startswith(b'PQHISTO'):
            self.file_type = PqFileType.PHU
        else:
            raise ValueError("Unsupported file type")

        self.version = f.read(8).rstrip(b'\0').decode('ascii', errors='ignore')

        while True:
            block = f.read(40)
            if len(block) < 40:
                raise ValueError("Incomplete tag block")
            tag_id_raw, index, typecode = struct.unpack('<32siI', block)
            tag_id = tag_id_raw.strip(b'\0').decode(errors='ignore')
            if tag_id == 'Header_End':
                self.data_offset = f.tell()
                break
            if index >= 0:
                tag_key = f"{tag_id}[{index}]"
            else:
                tag_key = tag_id
            tag_value = read_tag_value(f,typecode)
            self.tags[tag_key] = tag_value

        self.record_type = self.tags.get("TTResultFormat_TTTRRecType", None)
        self.bits_per_record = self.tags.get("TTResultFormat_BitsPerRecord", 32)

    def get(self, tag: str, default=None):
        return self.tags.get(tag, default)
    

    def read_records(self, count=None):
        dtype = np.uint32 if self.bits_per_record == 32 else np.uint64
        with open(self.path, 'rb') as f:
            f.seek(self.data_offset)
            return np.fromfile(f, dtype=dtype, count=count)

