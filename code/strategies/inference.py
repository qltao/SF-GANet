import argparse
import json
from pathlib import Path
import numpy as np
import rasterio
import torch
import torch.nn.functional as functional
from rasterio.windows import Window
from tqdm import tqdm

from model.model import SF_GANet
from strategies.run_strategy_comparison import strategy_config
from train.data_handler import calculate_indices_from_array
from train.training import load_model_checkpoint


def normalize_image_tile(tile_array, percentiles):
    normalized_tile = np.zeros_like(tile_array, dtype=np.float32)
    for band_index, (lower_bound, upper_bound) in enumerate(percentiles):
        if upper_bound > lower_bound:
            normalized_tile[band_index] = np.clip(
                (
                    tile_array[band_index].astype(np.float32)
                    - lower_bound
                )
                / (upper_bound - lower_bound),
                0,
                1,
            )
    normalized_tile[7:] = np.clip(
        (tile_array[7:].astype(np.float32) + 1) / 2,
        0,
        1,
    )
    return normalized_tile


def model_logits(model_output):
    if isinstance(model_output, tuple):
        return model_output[-1]
    return model_output


def predict_tile(model, tile_tensor, experiment_config):
    model.eval()
    window_size = experiment_config["data"]["window_size"]
    stride = max(
        1,
        int(
            round(
                window_size
                * (1 - experiment_config["data"]["overlap"])
            )
        ),
    )
    batch_size = experiment_config["training"]["batch_size"]
    device = experiment_config["training"]["device"]
    num_classes = experiment_config["data"]["num_classes"]

    _, height, width = tile_tensor.shape
    padding = max(1, window_size // 2)
    padded_tile = functional.pad(
        tile_tensor.unsqueeze(0),
        (padding, padding, padding, padding),
        mode="reflect",
    ).squeeze(0)

    _, padded_height, padded_width = padded_tile.shape
    probability_map = np.zeros(
        (num_classes, padded_height, padded_width),
        dtype=np.float32,
    )
    divisor_map = np.zeros(
        (padded_height, padded_width),
        dtype=np.float32,
    )
    patches, coordinates = [], []

    def process_batch():
        if not patches:
            return
        batch = torch.stack(patches).to(device)
        with torch.no_grad():
            probabilities = torch.softmax(
                model_logits(model(batch)),
                dim=1,
            ).cpu().numpy()
        for patch_index, (row_start, column_start) in enumerate(
            coordinates
        ):
            row_slice = slice(
                row_start, row_start + window_size
            )
            column_slice = slice(
                column_start, column_start + window_size
            )
            probability_map[:, row_slice, column_slice] += probabilities[
                patch_index, :, None, None
            ]
            divisor_map[row_slice, column_slice] += 1
        patches.clear()
        coordinates.clear()

    row_starts = list(
        range(
            0,
            padded_height - window_size + 1,
            stride,
        )
    )
    column_starts = list(
        range(
            0,
            padded_width - window_size + 1,
            stride,
        )
    )
    if row_starts[-1] != padded_height - window_size:
        row_starts.append(padded_height - window_size)
    if column_starts[-1] != padded_width - window_size:
        column_starts.append(padded_width - window_size)

    for row_start in row_starts:
        for column_start in column_starts:
            patches.append(
                padded_tile[
                    :,
                    row_start : row_start + window_size,
                    column_start : column_start + window_size,
                ]
            )
            coordinates.append((row_start, column_start))
            if len(patches) >= batch_size:
                process_batch()

    process_batch()
    divisor_map[divisor_map == 0] = 1
    probability_map /= divisor_map
    probability_map = probability_map[
        :,
        padding : padding + height,
        padding : padding + width,
    ]
    return np.argmax(probability_map, axis=0).astype(np.uint8) + 1


def run_checkpoint_inference(
    year,
    checkpoint_path,
    percentiles_path,
    output_path,
    experiment_config,
):
    checkpoint_path = Path(checkpoint_path)
    percentiles_path = Path(percentiles_path)
    output_path = Path(output_path)

    image_path = (
        experiment_config["paths"]["image_root"]
        / experiment_config["data"]["image_filename_format"].format(
            year=year
        )
    )
    if not image_path.is_file():
        raise FileNotFoundError(f"Input image not found: {image_path}")

    metadata = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )
    saved_config = metadata.get("run_config") or experiment_config
    model = SF_GANet(
        saved_config["data"]["num_classes"],
        9,
        pretrained=False,
        supcon=saved_config["experiment"]["use_supcon"],
    ).to(experiment_config["training"]["device"])
    load_model_checkpoint(
        model,
        checkpoint_path,
        experiment_config["training"]["device"],
    )
    percentiles = np.load(percentiles_path, allow_pickle=False)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(
        f"{output_path.stem}.partial{output_path.suffix}"
    )
    tile_size = experiment_config["inference"]["tile_size"]

    with rasterio.open(image_path) as source:
        profile = source.profile.copy()
        profile.update(dtype=rasterio.uint8, count=1, nodata=0)

        with rasterio.open(temporary, "w", **profile) as destination:
            total_rows = (
                source.height + tile_size - 1
            ) // tile_size
            total_columns = (
                source.width + tile_size - 1
            ) // tile_size

            for tile_row in tqdm(
                range(total_rows),
                desc=f"{year}",
                leave=False,
            ):
                for tile_column in range(total_columns):
                    column_offset = tile_column * tile_size
                    row_offset = tile_row * tile_size
                    width = min(
                        tile_size,
                        source.width - column_offset,
                    )
                    height = min(
                        tile_size,
                        source.height - row_offset,
                    )
                    window = Window(
                        column_offset,
                        row_offset,
                        width,
                        height,
                    )
                    original_bands = source.read(window=window)
                    feature_tile = np.vstack(
                        [
                            original_bands,
                            calculate_indices_from_array(
                                original_bands
                            ),
                        ]
                    )
                    normalized_tile = normalize_image_tile(
                        feature_tile,
                        percentiles,
                    )
                    predictions = predict_tile(
                        model,
                        torch.from_numpy(normalized_tile),
                        experiment_config,
                    )
                    destination.write(
                        predictions,
                        indexes=1,
                        window=window,
                    )

    temporary.replace(output_path)
    return output_path


def default_run_dir(strategy_name, experiment_config):
    seed = experiment_config["inference"]["map_seed"]
    return (
        experiment_config["paths"]["output_root"]
        / f"training_strategy_{strategy_name}"
        / experiment_config["evaluation"]["output_mode"]
        / f"training_strategy_{strategy_name}_seed{seed}"
    )


def run_strategy_inference(
    strategy_name,
    run_dir=None,
    year=None,
):
    experiment_config, _ = strategy_config(
        strategy_name,
        seed=42,
    )
    if run_dir is None:
        run_dir = default_run_dir(
            strategy_name,
            experiment_config,
        )
    run_dir = Path(run_dir)
    index_path = run_dir / "final_models" / "index.json"
    if not index_path.is_file():
        raise FileNotFoundError(
            f"Final model index not found: {index_path}"
        )

    index = json.loads(
        index_path.read_text(encoding="utf-8")
    )
    artifacts = {
        int(item["year"]): item
        for item in index["artifacts"]
    }

    configured_years = experiment_config["data"]["years"]
    years = [year] if year is not None else configured_years
    if any(value not in artifacts for value in years):
        raise ValueError("Requested year is missing from final_models.")

    output_root = run_dir / "maps" / "raw_30m"
    for current_year in years:
        item = artifacts[current_year]
        output = (
            output_root
            / f"LCZ_Map_{strategy_name}_{current_year}_raw_30m.tif"
        )
        run_checkpoint_inference(
            current_year,
            item["checkpoint_path"],
            item["percentiles_path"],
            output,
            experiment_config,
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--strategy",
        required=True,
        choices=["s1", "s2", "s3"],
    )
    parser.add_argument("--run_dir")
    parser.add_argument("--year", type=int)
    args = parser.parse_args()

    run_strategy_inference(
        args.strategy,
        args.run_dir,
        args.year,
    )


if __name__ == "__main__":
    main()
