# ITV Speech Classification Number

## Install
- B1: Sử dụng minconda khởi tạo 1 python env với python version = 3.10.
1. `conda create -n cls_num_env python=3.10 -y`
- B2: Chạy lệnh sau để kích hoạt môi trường tsd_env.
2. `conda activate tsd_env`
- B3: cài đặt các thư viện cần thiết cho dự án.
3. `pip install -r requirements.txt`

## Dữ liệu

## Train
- B1: Chỉnh sửa các tham số training và đường dẫn data trong file *./configs/default.yaml*
- B2: Chỉnh sửa `pre_trained_model` trong *./configs/default.yaml*. Giá trị này có thể điền theo 2 cách:
    - Tên của pretrain model theo format của huggingface. VD: *vinai/PhoWhisper-tiny*
    - Đường dẫn đến pretrain model folder trên local. VD: *./exp/models/best"*
- B3: Gõ lệnh sau để bắt đầu quá trình train
```python main.py train --config default.yaml --outfile log.log```
Mô hình tốt nhất được lưu tại *./exp/model/best*

Trong đó:
--config: đường dẫn đến file config chứa các tham số của mô hình. Lưu ý, chỉ tên file, không phải đường dẫn. Mặc định: “default.yaml”
--outfile: đường dẫn đến file log để lưu lại output của quá trình training. Lưu ý, chỉ tên file, không phải đường dẫn. Mặc định: “log.log””
