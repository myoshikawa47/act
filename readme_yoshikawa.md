### 環境構築

```
cd ~/work/2026/Turner/act
python -m venv .venv
source .venv/bin/activate
pip install torchvision
pip install torch
pip install pyquaternion
pip install pyyaml
pip install rospkg
pip install pexpect
pip install mujoco==2.3.7
pip install dm_control==1.0.14
pip install opencv-python
pip install matplotlib
pip install einops
pip install packaging
pip install h5py
pip install ipython
cd act/src/detr && pip install -e .
```

### dataset_dir の構造
データセットは `dataset_dir` 直下に episode 単位の `npz` を配置します。

```text
dataset_dir/
├── episode1.npz
├── episode2.npz
├── ...
```

各 `npz` は少なくとも以下のキーを持つ前提です。

- 画像
  - `head_right_camera`
  - `head_left_camera`
  - `right_hand_camera`
  - `left_hand_camera`
- 関節 command
  - `left_arm_command`
  - `right_arm_command`

画像は JPEG/PNG 圧縮済みバイト列の `object` 配列、command は `float32` の時系列配列です。

### 現状のデータ処理フロー
`act/src/utils.py` の `load_data()` が学習用データを構築します。

1. `dataset_dir` 直下の `*.npz` を列挙する。
2. `train_ratio` と `shuffle` に従って train / val に split する。
3. `get_norm_stats()` で全 episode の `qpos` 統計を計算する。
   - `qpos = concat(left_arm_command, right_arm_command)`
   - `action` の統計は `qpos` の統計をそのまま流用する。
4. 全 episode を見て、使用カメラと左右 command の共通最短長から `max_episode_len` を決める。
5. `EpisodicDataset` が各 episode を初期化時にロードする。
   - episode ごとの有効長は、左右 command と使用カメラ列の長さの最小値。
   - 画像は全フレームを一度だけ `cv2.imdecode()` + `cv2.resize()` してメモリに保持する。
6. `__getitem__()` では 1 episode から 1 時刻 `start_ts` をランダムに選ぶ。
   - 観測 `qpos` は `qpos[start_ts]`
   - 観測画像は各カメラの `image[start_ts]`
   - 教師 `action` は `qpos[start_ts + 1:]`
7. `action` はデータセット全体共通の `max_episode_len` までゼロパディングする。
   - `is_pad=False` が有効 action 区間
   - `is_pad=True` が pad 区間
8. 返り値は以下。
   - `image_data`: `(num_cameras, 3, camera_height, camera_width)`
   - `qpos_data`: `(state_dim,)`
   - `action_data`: `(max_episode_len, state_dim)`
   - `is_pad`: `(max_episode_len,)`

### action / qpos の仕様
- `qpos` は `left_arm_command` と `right_arm_command` の結合です。
- `state_dim` は現在 `18` を想定しています。
- `action` は別データではなく、同じ `qpos` を 1 step 未来にずらしたものを使います。
  - `action[t] = qpos[t + 1]`
- 最終時刻には未来の `qpos` がないため、episode 末尾側は pad されます。

### 画像処理の仕様
- 画像は `npz` 内では圧縮バイト列です。
- `act/src/utils.py` の `_decode_image()` で `cv2.imdecode()` して RGB に変換します。
- その後 `cv2.resize(image, (camera_width, camera_height))` でリサイズします。
- 初期化時に全フレームを decode 済みにするため、学習中の `__getitem__()` では追加 decode は発生しません。
- 異なる元解像度のカメラでも、学習入力サイズは揃います。

### train / val split の仕様
`act/src/utils.py` の `split_indices()` / `load_data()` で split します。

- `train_ratio`: `train : val = train_ratio : 1 - train_ratio`
- `shuffle=True`: seed 付きランダム順で split
- `shuffle=False`: ファイル名ソート順の先頭から train, 残りを val
- `act/src/test.py` も同じ split ロジックを使います。
  - `--mode train` で train split
  - `--mode val` または `--mode test` で val split

### DataLoader の仕様
- `num_workers = min(8, max(1, os.cpu_count() // 2))`
- `prefetch_factor = 2`
- `pin_memory = True`

### data-specific に修正が必要な箇所
#### `act/src/imitate_episodes.py`
- `main()` 内の `camera_names`
  - 現状は `['head_right_camera', 'left_hand_camera']`
  - 学習に使うカメラに合わせて修正する。
- parser 引数の default
  - `dataset_dir`
  - `episode_len`
  - `state_dim`
  - `camera_height`
  - `camera_width`
  - `train_ratio`

#### `act/src/utils.py`
- `LEFT_QPOS_KEY`, `RIGHT_QPOS_KEY`
  - 現状は `left_arm_command`, `right_arm_command`
- `load_npz_episode()`
  - 使用するカメラキー名
  - 画像圧縮形式
- `_get_episode_len_from_arrays()`
  - 使用カメラと state の長さ整合条件
- `_decode_image()`
  - 圧縮形式や前処理が変わる場合

#### `act/src/test.py`
- `mode` と split の扱い
- 出力画像 / 動画の保存先
- 学習時と同じ `camera_names`, `camera_height`, `camera_width`, `train_ratio`, `shuffle`, `seed` を `config.yaml` から読む

### 主な引数
`act/src/imitate_episodes.py` の parser で受ける主な引数は以下です。

- `--dataset_dir`
  - `npz` データセットのルートディレクトリ
- `--episode_len`
  - 設定保存用。実データ長は `utils.py` 側で `npz` から決まる
- `--state_dim`
  - 現状は `18`
- `--batch_size`
  - train / val 共通 batch size
- `--train_ratio`
  - train split の比率
- `--shuffle`, `--no_shuffle`
  - split 時に episode 順をシャッフルするか
- `--camera_height`
  - デコード後画像の高さ
- `--camera_width`
  - デコード後画像の幅
- `--device`
  - `0,1,...` で GPU、`-1` で CPU
- `--temporal_agg`
  - 時間方向アグリゲーションを有効化
- `--policy_class`
  - `ACT` または `CNNMLP`
- `--chunk_size`
  - ACT の query 長
- `--hidden_dim`
- `--dim_feedforward`
- `--kl_weight`
- `--lr`
- `--num_epochs`
- `--seed`
- `--ckpt_dir`

### 学習コマンド例
```bash
python3 ./src/imitate_episodes.py \
  --dataset_dir /home/tnakagawa/work/2026/Turner/dataset/npz \
  --temporal_agg \
  --device 0
```

### オフラインテスト例
```bash
python3 ./src/test.py \
  --ckpt_path ./log/<timestamp>/policy_best.ckpt \
  --mode val \
  --idx 0 \
  --device 0 \
  --output image
```

### 現時点の注意点
- `episode_len` 引数は設定ファイル保存には使われますが、データローダの実長制御は `npz` 側の時系列長で決まります。
- 現状の `camera_names` はコード内固定なので、データセットに合わせて変更が必要です。
- 画像を全フレーム事前 decode するので、データ量が増えるとメモリ使用量も増えます。
- GPU 利用時は PyTorch と NVIDIA driver の CUDA 互換性が必要です。
