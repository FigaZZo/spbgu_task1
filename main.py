#!/usr/bin/env python3

from __future__ import annotations

import argparse
import hashlib
import ipaddress
import os
import sys
import tempfile
from dataclasses import dataclass
from typing import List, Tuple, Optional


RECORD_SIZE = 16  # IPv6 packed = 16 bytes


def parse_ipv6_to_packed(s: str) -> bytes:
    """
    Канонизация IPv6:
    - ipaddress.IPv6Address принимает и сокращённую форму, и любые регистры
    - .packed - 16 байт, одинаковые для всех текстовых вариаций одного адреса
    """
    return ipaddress.IPv6Address(s.strip()).packed


def fast_bucket_index(packed: bytes, mask: int, shift: int) -> int:
    """
    Получаем индекс бакетa из хеша
    Используем blake2b, 8 байт достаточно
    """
    h = hashlib.blake2b(packed, digest_size=8).digest()
    x = int.from_bytes(h, "little")
    return (x >> shift) & mask


@dataclass
class BucketingPlan:
    bits: int  # число бит под бакет => bucket_count = 2**bits

    @property
    def bucket_count(self) -> int:
        return 1 << self.bits

    @property
    def mask(self) -> int:
        return (1 << self.bits) - 1


def estimate_default_bits() -> int:
    """
    Безопасный дефолт: 2^12 бакет
    Это обычно хорошо снижает пик памяти
    """
    return 12


def count_unique_in_memory(input_path: str) -> int:
    """
    Базовое решение. хранение всех уникальных адресов в RAM
    """
    seen = set()
    with open(input_path, "rt", encoding="utf-8", errors="strict") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            seen.add(parse_ipv6_to_packed(line))
    return len(seen)


def write_buckets_level(
    input_path: str,
    temp_dir: str,
    plan: BucketingPlan,
    prefix: str,
) -> List[str]:
    """
    Разбивает входной текстовый файл на бакет файлы
    Возвращает список путей бакет-файлов
    """
    bucket_paths = [
        os.path.join(temp_dir, f"{prefix}_b{i:0{len(str(plan.bucket_count))}d}.bin")
        for i in range(plan.bucket_count)
    ]

    """
    Открываем все бакет файлы сразу: быстрее, но держит дескрипторы
    При 4096 бакет может быть много. Поэтому открываем батчами, но сделаем 
    кэш открытых файлов с ограничением
    """
    max_open = 256
    open_files = {}
    order = []

    def get_handle(idx: int):
        path = bucket_paths[idx]
        if idx in open_files:
            return open_files[idx]
        # LRU закрытие
        if len(open_files) >= max_open:
            old = order.pop(0)
            open_files[old].close()
            del open_files[old]
        fh = open(path, "ab", buffering=1024 * 1024)
        open_files[idx] = fh
        order.append(idx)
        return fh

    mask = plan.mask
    shift = 0  # в первом уровне берём младшие bits из хеша

    with open(input_path, "rt", encoding="utf-8", errors="strict") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            packed = parse_ipv6_to_packed(line)
            idx = fast_bucket_index(packed, mask=mask, shift=shift)
            get_handle(idx).write(packed)

    for fh in open_files.values():
        fh.close()

    return bucket_paths


def write_buckets_level_from_bin(
    bin_path: str,
    temp_dir: str,
    plan: BucketingPlan,
    prefix: str,
    shift: int,
) -> List[str]:
    """
    Второй уровень разбиения: исходник уже бинарный (16 байт/запись)
    Разбиваем по другим битам хеша (shift>0), чтобы рассеять "тяжёлый" бакет
    """
    bucket_paths = [
        os.path.join(temp_dir, f"{prefix}_b{i:0{len(str(plan.bucket_count))}d}.bin")
        for i in range(plan.bucket_count)
    ]

    max_open = 256
    open_files = {}
    order = []

    def get_handle(idx: int):
        path = bucket_paths[idx]
        if idx in open_files:
            return open_files[idx]
        if len(open_files) >= max_open:
            old = order.pop(0)
            open_files[old].close()
            del open_files[old]
        fh = open(path, "ab", buffering=1024 * 1024)
        open_files[idx] = fh
        order.append(idx)
        return fh

    mask = plan.mask

    with open(bin_path, "rb") as f:
        while True:
            rec = f.read(RECORD_SIZE)
            if not rec:
                break
            if len(rec) != RECORD_SIZE:
                raise RuntimeError(f"Corrupt bin record in {bin_path}")
            idx = fast_bucket_index(rec, mask=mask, shift=shift)
            get_handle(idx).write(rec)

    for fh in open_files.values():
        fh.close()

    return bucket_paths


def count_unique_in_bin_bucket(bin_path: str, memory_limit_bytes: int) -> int:
    """
    Считает уникальные в одном bin бакет файле.
    Если бакет слишком большой для memory_limit_bytes (грубая оценка),
    возвращает -1, чтобы вызывающий код сделал вторичный сплит.
    """
    size = os.path.getsize(bin_path)
    if size == 0:
        return 0

    # Грубая оценка: set из bytes имеет большой overhead.
    # В реальности 16-байтные элементы в set могут потреблять ~70-100+ байт.
    # Возьмём консервативно: 96 байт на запись.
    approx_per_item = 96
    est_items = size // RECORD_SIZE
    est_mem = est_items * approx_per_item

    if est_mem > memory_limit_bytes:
        return -1

    seen = set()
    with open(bin_path, "rb") as f:
        while True:
            rec = f.read(RECORD_SIZE)
            if not rec:
                break
            seen.add(rec)
    return len(seen)


def count_unique_external(
    input_path: str,
    memory_limit_bytes: int = 900 * 1024 * 1024,  # оставляем запас < 1 ГБ
    bits_level1: int = 12,  # 4096
    bits_level2: int = 10,  # 1024
) -> int:
    """
    Оптимизированное решение:
    - 1-й уровень  по хешу - много небольших bin-файлов
    - суммируем уникальные по бакетам
    - если бакет слишком большой - 2-й уровень разбиения и считаем по под-бакетам
    """
    plan1 = BucketingPlan(bits=bits_level1)
    plan2 = BucketingPlan(bits=bits_level2)

    total = 0
    with tempfile.TemporaryDirectory(prefix="ipv6uniq_") as tmp:
        buckets = write_buckets_level(input_path, tmp, plan1, prefix="L1")

        for bpath in buckets:
            if not os.path.exists(bpath):
                continue
            if os.path.getsize(bpath) == 0:
                continue

            cnt = count_unique_in_bin_bucket(bpath, memory_limit_bytes=memory_limit_bytes)
            if cnt >= 0:
                total += cnt
                continue

            # Слишком большой bucket — делаем 2-й уровень.
            # shift = bits_level1: берём следующие биты хеша, чтобы разбить иначе.
            sub_buckets = write_buckets_level_from_bin(
                bpath, tmp, plan2, prefix=os.path.basename(bpath) + "_L2", shift=bits_level1
            )

            for sb in sub_buckets:
                if os.path.getsize(sb) == 0:
                    continue
                scnt = count_unique_in_bin_bucket(sb, memory_limit_bytes=memory_limit_bytes)
                if scnt < 0:
                    # На практике это крайне редко; можно добавить 3-й уровень,
                    # но в рамках олимпиады разумно сигнализировать.
                    raise MemoryError(
                        f"Bucket still too large after level-2 split: {sb}. "
                        f"Consider increasing bits_level2 or adding level-3."
                    )
                total += scnt

    return total


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Count unique IPv6 addresses in a file (exact), memory-safe."
    )
    parser.add_argument("input_path", help="Path to input text file with IPv6 addresses")
    parser.add_argument("output_path", help="Path to output file (single integer)")
    parser.add_argument(
        "--mode",
        choices=["auto", "mem", "ext"],
        default="auto",
        help="mem: in-memory set; ext: external bucketing; auto: choose by file size",
    )
    parser.add_argument(
        "--auto-threshold-mb",
        type=int,
        default=200,
        help="If input file <= this size (MB), use in-memory mode in auto",
    )
    parser.add_argument(
        "--memory-limit-mb",
        type=int,
        default=900,
        help="Approx RAM limit for external mode (MB). Keep below ~1024",
    )
    parser.add_argument(
        "--bits-level1",
        type=int,
        default=estimate_default_bits(),
        help="Bucket bits for level-1 split (2^bits buckets). Default 12 => 4096.",
    )
    parser.add_argument(
        "--bits-level2",
        type=int,
        default=10,
        help="Bucket bits for level-2 split for heavy buckets. Default 10 => 1024.",
    )

    args = parser.parse_args(argv)

    input_size = os.path.getsize(args.input_path)
    threshold = args.auto_threshold_mb * 1024 * 1024

    if args.mode == "mem" or (args.mode == "auto" and input_size <= threshold):
        result = count_unique_in_memory(args.input_path)
    else:
        result = count_unique_external(
            args.input_path,
            memory_limit_bytes=args.memory_limit_mb * 1024 * 1024,
            bits_level1=args.bits_level1,
            bits_level2=args.bits_level2,
        )

    with open(args.output_path, "wt", encoding="utf-8") as out:
        out.write(str(result))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
