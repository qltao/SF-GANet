import json
from pathlib import Path
import numpy as np
import rasterio
import torch
import torch.nn.functional as functional
from rasterio.windows import Window
from tqdm import tqdm
from train.training import load_model_checkpoint
from train.data_handler import calculate_indices_from_array
from model.model import SF_GANet
import argparse
from strategies.run_strategy_comparison import strategy_config


def normalize_image_tile(tile_array, percentiles):
    normalized_tile = np.zeros_like(tile_array, dtype=np.float32)
    for band_index, (lower_bound, upper_bound) in enumerate(percentiles):
        if upper_bound > lower_bound:
            normalized_tile[band_index] = np.clip(
                (tile_array[band_index].astype(np.float32) - lower_bound) / (upper_bound - lower_bound),
                0,
                1,
            )
    normalized_tile[7:] = np.clip((tile_array[7:].astype(np.float32) + 1) / 2, 0, 1)
    return normalized_tile


def split_model_output(model_output):
    if isinstance(model_output, tuple):
        return model_output[-1]
    return model_output


def predict_tile(model, tile_tensor, experiment_config):
    model.eval()
    window_size = experiment_config["data"]["window_size"]
    stride = max(1, int(round(window_size * (1 - experiment_config["data"]["overlap"]))))
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
    probability_map = np.zeros((num_classes, padded_height, padded_width), dtype=np.float32)
    divisor_map = np.zeros((padded_height, padded_width), dtype=np.float32)
    patches, coordinates = [], []

    def process_batch():
        if not patches:
            return
        batch = torch.stack(patches).to(device)
        with torch.no_grad():
            logits = split_model_output(model(batch))
            probabilities = torch.softmax(logits, dim=1).cpu().numpy()
        for patch_index, (row_start, column_start) in enumerate(coordinates):
            row_slice = slice(row_start, row_start + window_size)
            column_slice = slice(column_start, column_start + window_size)
            probability_map[:, row_slice, column_slice] += probabilities[
                patch_index,
                :,
                None,
                None,
            ]
            divisor_map[row_slice, column_slice] += 1
        patches.clear()
        coordinates.clear()

    row_starts = list(range(0, padded_height - window_size + 1, stride))
    column_starts = list(range(0, padded_width - window_size + 1, stride))
    if row_starts[-1] != padded_height - window_size:
        row_starts.append(padded_height - window_size)
    if column_starts[-1] != padded_width - window_size:
        column_starts.append(padded_width - window_size)
    for row_start in row_starts:
        for column_start in column_starts:
            patches.append(padded_tile[:, row_start:row_start + window_size, column_start:column_start + window_size])
            coordinates.append((row_start, column_start))
            if len(patches) >= batch_size:
                process_batch()
    process_batch()
    divisor_map[divisor_map == 0] = 1
    probability_map /= divisor_map
    probability_map = probability_map[:, padding:padding + height, padding:padding + width]
    return np.argmax(probability_map, axis=0).astype(np.uint8) + 1


def inspect_map_output(output_path, expected_metadata):
    output_path = Path(output_path)
    metadata_path = output_path.with_suffix('.json')
    if not output_path.exists() and not metadata_path.exists():
        return 'missing'
    if not output_path.exists() or not metadata_path.exists():
        return 'incomplete'
    try:
        saved_metadata = json.loads(metadata_path.read_text(encoding='utf-8'))
    except (OSError, json.JSONDecodeError):
        return 'incomplete'

    path_keys = ['run_dir', 'checkpoint', 'percentiles']
    for key in path_keys:
        saved_value = saved_metadata.get(key)
        expected_value = expected_metadata.get(key)
        if saved_value is None or expected_value is None:
            return 'incomplete'
    for key in ['year', 'scope', 'selected_fold']:
        if key in expected_metadata and saved_metadata.get(key) != expected_metadata[key]:
            raise FileExistsError(
                f'地图来源字段{key}不一致，拒绝覆盖：{output_path}'
            )
    try:
        with rasterio.open(output_path) as source:
            if source.count != 1 or source.width < 1 or source.height < 1:
                return 'incomplete'
    except (OSError, rasterio.errors.RasterioError):
        return 'incomplete'
    return 'complete'


def write_map_metadata(output_path, metadata):
    metadata_path = Path(output_path).with_suffix('.json')
    partial_path = metadata_path.with_name(f'{metadata_path.name}.partial')
    completed_metadata = dict(metadata, completed=True)
    partial_path.write_text(
        json.dumps(completed_metadata, ensure_ascii=False, indent=2),
        encoding='utf-8',
    )
    partial_path.replace(metadata_path)


def run_checkpoint_inference(
    year,
    model_name,
    checkpoint_path,
    percentiles_path,
    output_path,
    experiment_config,
):
    checkpoint_path = Path(checkpoint_path)
    percentiles_path = Path(percentiles_path)
    output_path = Path(output_path)
    partial_output_path = output_path.with_name(
        f'{output_path.stem}.partial{output_path.suffix}'
    )
    image_path = (
        experiment_config["paths"]["image_root"]
        / experiment_config["data"]["image_filename_format"].format(year=year)
    )

    metadata = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    saved_config = metadata.get('run_config') or experiment_config
    model = SF_GANet(
        saved_config["data"]["num_classes"], 9,
        pretrained=False, supcon=saved_config['experiment']['use_supcon'],
    ).to(experiment_config["training"]["device"])
    load_model_checkpoint(model, checkpoint_path, experiment_config["training"]["device"])
    percentiles = np.load(percentiles_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tile_size = experiment_config["inference"]["tile_size"]

    with rasterio.open(image_path) as source:
        output_profile = source.profile.copy()
        output_profile.update(dtype=rasterio.uint8, count=1, nodata=0)
        with rasterio.open(partial_output_path, "w", **output_profile) as destination:
            total_rows = (source.height + tile_size - 1) // tile_size
            total_columns = (source.width + tile_size - 1) // tile_size
            with tqdm(
                total=total_rows * total_columns,
                desc=f"{year} tile",
                unit="tile",
                dynamic_ncols=True,
                leave=False,
            ) as progress_bar:
                for tile_row in range(total_rows):
                    for tile_column in range(total_columns):
                        column_offset = tile_column * tile_size
                        row_offset = tile_row * tile_size
                        width = min(tile_size, source.width - column_offset)
                        height = min(tile_size, source.height - row_offset)
                        window = Window(column_offset, row_offset, width, height)
                        original_bands = source.read(window=window)
                        feature_tile = np.vstack([
                            original_bands,
                            calculate_indices_from_array(original_bands),
                        ])
                        normalized_tile = normalize_image_tile(feature_tile, percentiles)
                        predictions = predict_tile(
                            model,
                            torch.from_numpy(normalized_tile),
                            experiment_config,
                        )
                        destination.write(predictions, indexes=1, window=window)
                        progress_bar.update(1)
    partial_output_path.replace(output_path)
    print(f" {year} output to：{output_path}")
    return output_path


map_strategies = ("s1", "s2", "s3")


def run_strategy_inference(strategy_name, years, selection_dir, check_only=False):
    experiment_config, _ = strategy_config(strategy_name)
    output_root = Path(selection_dir).resolve()
    binding = json.loads((output_root / 'selection.json').read_text(encoding='utf-8'))
    source = Path(binding['run_dir'])
    print(f'select {binding["selected_fold"]}', flush=True)
    year_progress = tqdm(
        years,
        desc=f'{strategy_name.upper()}',
        unit='年',
        dynamic_ncols=True,
    )
    for year in year_progress:
        year_progress.set_postfix_str(f'{year}')
        matches = [item for item in binding['artifacts'] if item['year'] == year]
        checkpoint, percentiles = matches[0]['checkpoint_path'], matches[0]['percentiles_path']
        print(f'INFERENCE_SOURCE {strategy_name}/{year}: checkpoint={checkpoint} '
              f'percentiles={percentiles}', flush=True)
        if check_only:
            continue
        output = output_root / 'maps' / 'raw' / f'LCZ_Map_{strategy_name}_{year}_raw_10m.tif'
        metadata = {
            'run_dir': str(source), 'checkpoint': str(checkpoint),
            'percentiles': str(percentiles), 'year': year,
            'scope': 'strategy_selected_fold_map_not_independent_accuracy',
            'selected_fold': binding['selected_fold'],
        }
        output_status = inspect_map_output(output, metadata)
        if output_status == 'complete':
            print(f'年份 {year} 已完整完成，跳过：{output}', flush=True)
            continue
        if output_status == 'incomplete':
            print(f'年份 {year} 存在中断产物，从该年开头重新推理。', flush=True)
        run_checkpoint_inference(
            year, 'sf_ganet', checkpoint, percentiles, output, experiment_config,
        )
        # 单独保存本幅地图来源，不修改已完成运行的训练配置。
        write_map_metadata(output, metadata)


def main():
    parser = argparse.ArgumentParser(description='使用已选年度权重进行策略地图预测')
    parser.add_argument('--strategy', required=True, choices=map_strategies)
    parser.add_argument('--selection_dir', required=True,
                        help='原实验 strategy_best/<选择哈希> 目录，内有 selection.json')
    parser.add_argument('--year', type=int)
    parser.add_argument('--check_only', action='store_true')
    arguments = parser.parse_args()
    experiment_config, _ = strategy_config(arguments.strategy)
    configured_years = experiment_config['data']['years']
    years = [arguments.year] if arguments.year is not None else configured_years
    if any(year not in configured_years for year in years):
        raise ValueError('年份不在配置范围内。')
    run_strategy_inference(
        arguments.strategy, years, arguments.selection_dir, arguments.check_only,
    )


if __name__ == '__main__':
    main()
