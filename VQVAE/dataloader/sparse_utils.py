from pathlib import Path, PurePosixPath

import torch


AMASS_SUBSETS = {
    "ACCAD",
    "BMLmovi",
    "BioMotionLab_NTroje",
    "CMU",
    "EKUT",
    "Eyes_Japan_Dataset",
    "HumanEva",
    "KIT",
    "MPI_HDM05",
    "MPI_Limits",
    "MPI_mosh",
    "SFU",
    "TotalCapture",
    "Transitions_mocap",
}

PROTOCOL_SPLITS = {
    "1_small_dataset": {
        "train": ("CMU_train.txt", "BioMotionLab_NTroje_train.txt", "MPI_HDM05_train.txt"),
        "val": ("CMU_test.txt", "BioMotionLab_NTroje_test.txt", "MPI_HDM05_test.txt"),
        "test": ("CMU_test.txt", "BioMotionLab_NTroje_test.txt", "MPI_HDM05_test.txt"),
    },
    "randomsplit_0": {
        "train": ("s3_train.txt",),
        "val": ("s3_test.txt",),
        "test": ("s3_test.txt",),
    },
}


def sparse_joint_indices(sparse_dim):
    sparse_dim = int(sparse_dim)
    if sparse_dim == 54:
        return [15, 20, 21]
    if sparse_dim == 72:
        return [0, 15, 20, 21]
    raise ValueError(f"Unsupported SPARSE_DIM={sparse_dim}. Expected 54 or 72.")


def sparse_head_hand_indices(sparse_dim):
    sparse_dim = int(sparse_dim)
    if sparse_dim == 54:
        return 0, [1, 2]
    if sparse_dim == 72:
        return 1, [2, 3]
    raise ValueError(f"Unsupported SPARSE_DIM={sparse_dim}. Expected 54 or 72.")


def normalize_source_path(value):
    raw = str(value).replace("\\", "/")
    parts = [part for part in PurePosixPath(raw).parts if part not in ("/", "")]
    for index, part in enumerate(parts):
        if part in AMASS_SUBSETS:
            relative = PurePosixPath(*parts[index:]).as_posix()
            if not relative.endswith(".npz"):
                raise ValueError(f"Expected an AMASS NPZ path, got: {value}")
            return relative
    raise ValueError(f"Cannot normalize AMASS source path: {value}")


def data_split_file(filename):
    repo_root = Path(__file__).resolve().parents[2]
    candidates = [
        repo_root / "VQVAE" / "prepare_data" / "data_split" / filename,
        Path(filename),
    ]
    for path in candidates:
        if path.is_file():
            return path
    raise FileNotFoundError(f"Cannot find split file {filename}; checked: {candidates}")


def protocol_split_files(protocol, split):
    if protocol not in PROTOCOL_SPLITS:
        raise ValueError(f"Unsupported protocol: {protocol}")
    if split not in PROTOCOL_SPLITS[protocol]:
        raise ValueError(f"Unsupported split {split!r} for protocol {protocol!r}")
    return [data_split_file(filename) for filename in PROTOCOL_SPLITS[protocol][split]]


def read_split_sources(protocol, split):
    split_files = protocol_split_files(protocol, split)
    sources = []
    seen = set()
    for split_file in split_files:
        for line_number, line in enumerate(split_file.read_text().splitlines(), start=1):
            if not line.strip():
                continue
            source = normalize_source_path(line.strip())
            if source in seen:
                raise ValueError(f"Duplicate source in {split_file}:{line_number}: {source}")
            seen.add(source)
            sources.append(source)
    return split_files, sources


def resolve_dataset_path(source_path, dataset_path):
    source = Path(normalize_source_path(source_path))
    if source.is_absolute():
        raise ValueError(f"Split entries must be relative paths: {source_path}")
    return Path(dataset_path).joinpath(*source.with_suffix(".pt").parts)


def source_from_processed_path(processed_path, dataset_path):
    processed = Path(processed_path).absolute()
    dataset_root = Path(dataset_path).absolute()
    try:
        relative = processed.relative_to(dataset_root)
    except ValueError as error:
        raise ValueError(f"Processed file is outside DATASET_PATH: {processed}") from error
    return relative.with_suffix(".npz").as_posix()


def load_processed_record(processed_path, dataset_path):
    processed_path = Path(processed_path)
    try:
        record = torch.load(processed_path, map_location="cpu", weights_only=False)
    except TypeError:
        record = torch.load(processed_path, map_location="cpu")

    if "filepath" not in record:
        raise KeyError(f"Missing filepath metadata in {processed_path}")
    expected = source_from_processed_path(processed_path, dataset_path)
    actual = normalize_source_path(record["filepath"])
    if actual != expected:
        raise ValueError(
            f"Processed/source mismatch for {processed_path}: expected {expected}, metadata says {actual}"
        )
    return record


def split_processed_paths(dataset_path, protocol, split):
    split_files, sources = read_split_sources(protocol, split)
    processed = [resolve_dataset_path(source, dataset_path) for source in sources]
    missing = [str(path) for path in processed if not path.is_file()]
    if missing:
        preview = "\n".join(missing[:20])
        raise FileNotFoundError(
            f"Missing {len(missing)} processed files for {split_files}:\n{preview}"
        )
    return split_files, sources, processed


def stable_sequence_name(source_path):
    return Path(normalize_source_path(source_path)).with_suffix("").as_posix()
