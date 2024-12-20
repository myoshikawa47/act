### dataset_dir の構造（データセットは正規化前の生データ）
```
dataset_dir/
├── train/
│   ├── cam1.npy          # (datanum, step, height, width, channel)
│   ├── cam2.npy          # (datanum, step, height, width, channel)
│   ├── robot_states.npy  # (datanum, step, state_dim)
│   └── ...               # (datanum, step, feature_dim)
└── test/
    ├── cam1.npy          # (datanum, step, height, width, channel)
    ├── cam2.npy          # (datanum, step, height, width, channel)
    ├── robot_states.npy  # (datanum, step, state_dim)
    └── ...               # (datanum, step, feature_dim)
```


### 修正が必要なパラメータ
- act/src/imitate_episodes.py  
    - in main() L67  
        - camera_names (=['cam1', 'cam2'])  
    - in L448~ parser  
        - dataset_dir  
        - episode_len  
        - state_dim  
            
- act/src/utils.py  
    - in get_norm_stats() L90  
        - npyファイル名 (='robot_states')  
    - in Class EpisodicDataset L26  
        - npyファイル名 (='robot_states')  

- act/src/test.py  
    - in L58  
        - npyファイル名 (='robot_states')  

### observationとactionには同一データを使用

### 学習
`(aloha) usr@rtx01:~/act/act$ python3 ./src/imitate_episode.py --temporal_agg --device 0`

### オフラインテスト
`(aloha) usr@rtx01:~/act/act$ python3 ./src/test.py --ckpt_path ./log/<tagname>/policy_best.ckpt --device 0`
