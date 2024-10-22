### dataset_dir の構造
- dataset_dir:  
    - train:  
        - cam1.npy (datanum, step, channel, height, width)  
        - cam2.npy  
        - robot_states.npy (datanum, step, state_dim)  
        - ...  
    - test:  
        - cam1.npy  
        - cam2.npy  
        - robot_states.npy  
        - ...  

### 修正が必要なパラメータ
- in imitate_episode.py  
    - in main() L67  
        - camera_names (=['cam1', 'cam2'])  
    - in L448~ parser  
        - dataset_dir  
        - episode_len  
        - state_dim  
            
- in utils.py  
    - in get_norm_stats() L90  
        - npy ファイル名 (='robot_states')  
    - in Class EpisodicDataset L26  
        - npy ファイル名 (='robot_states')  

### 注意：デフォルトではobservationとactionは同一のデータ

### 学習
`(aloha) usr@rtx01:~/act$ python3 ./src/imitate_episode.py --temporal_agg --device 0`

### オフラインテスト
`(aloha) usr@rtx01:~/act$ python3 ./src/test.py --ckpt_path ./log/'tagname'/policy_best.ckpt --device 0`